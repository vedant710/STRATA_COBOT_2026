#!/usr/bin/env python3

# Filename: task3B_planning.py
# Nodes:    ebot_planner_node
#           Publishing Topics  - [ /ebot_path, /handover/ebot_ready ]
#           Subscribing Topics - [ /map, /odom, /handover/package_taken, /ebot_nav/step_done ]
#
# Changes from Task 3A:
#   - a step ends when the navigation node reports it has stopped (/ebot_nav/step_done),
#     not when a noisy /odom twist looks still
#   - larger clearance from rocks; path shortcuts are checked against a bigger safety radius
#   - the node resets itself for a new run (new map, or the simulation clock jumping back),
#     so the same node serves two consecutive runs


import rclpy
import sys
import math
import heapq
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import Bool
from std_srvs.srv import Trigger


##################### TASK CONSTANTS #######################

map_topic = '/map'
path_topic = '/ebot_path'
odom_topic = '/odom'
spawn_service = '/spawn_ore_package'

ready_topic = '/handover/ebot_ready'
taken_topic = '/handover/package_taken'
done_topic = '/ebot_nav/step_done'      # nav -> planner: True when the step has finished

map_frame = 'map'

home_pose = (0.0, 0.0, 0.0)
ore_drop_pose = (6.0, 1.9, 0.0)

# TODO (MEASURE THIS): use the arm pose you measured in Task 3A
arm_pose = (0.0, 1.9, math.pi)

steps = [
    (home_pose, ore_drop_pose),
    (ore_drop_pose, arm_pose),
    (arm_pose, home_pose),
]
ARM_STEP = 1
LEAVE_STEP = 2

# ---- map reading ----
occ_threshold = 50
plan_factor = 5

# ---- start / end of a step ----
start_delay = 1.0           # s after a run starts before the first path is planned
end_settle = 0.4            # s the eBot must have been stopped (nav reported done)
pos_tol = 0.35              # loose check against the noisy /odom
yaw_tol = 0.25

# ---- planning (more clearance than Task 3A) ----
robot_radius = 0.40
margin_steps = (0.15, 0.12, 0.08, 0.05, 0.02, 0.0)   # tried in order until a path exists
smooth_extra = 0.08         # shortcuts must keep this much extra clearance
clear_radius = 0.30
near_band = 0.20
near_penalty = 2.0
path_spacing = 0.20

# ---- retry timing ----
plan_retry_s = 2.0
ore_retry_s = 1.0
wait_log_s = 5.0


##################### CLASS DEFINITION #######################

class ebot_planner(Node):
    '''
    Plans each leg, publishes it on /ebot_path, requests the ore package at the ore drop
    pose, tells the arm when the eBot has arrived, and holds the eBot beside the arm until
    the arm has taken the package. Resets itself for every new run.
    '''

    def __init__(self):
        super().__init__(
            'ebot_planner_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        ############ Topic SUBSCRIPTIONS ############

        self.map_sub = self.create_subscription(OccupancyGrid, map_topic, self.mapcb, latched)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)
        self.taken_sub = self.create_subscription(Bool, taken_topic, self.takencb, latched)
        self.done_sub = self.create_subscription(Bool, done_topic, self.nav_donecb, latched)

        ############ Topic PUBLISHERS ############

        self.path_pub = self.create_publisher(Path, path_topic, latched)
        self.ready_pub = self.create_publisher(Bool, ready_topic, latched)

        ############ Service CLIENTS ############

        self.spawn_client = self.create_client(Trigger, spawn_service)

        ############ Constructor VARIABLES/OBJECTS ############

        self.grid = None
        self.odom = None
        self.step = 0

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.path_published = False
        self.ore_requested = False
        self.ore_done = False
        self.finished = False
        self.next_plan_time = 0.0
        self.next_ore_time = 0.0
        self.next_wait_log = 0.0

        self.package_taken = False
        self.ready_sent = False

        self.nav_done = False
        self.nav_done_time = 0.0
        self.run_start_time = 0.0

        self.coarse = None
        self.coarse_resolution = None
        self.map_bytes = None
        self.last_now = 0.0

        self.publish_ready(False)

        plan_rate = 0.1
        self.timer = self.create_timer(plan_rate, self.process_planning)


    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9


    # ------------------------- run reset -------------------------

    def reset_run(self):
        '''Put the node back in its start-of-run state (called for a new map / new run).'''
        self.step = 0
        self.path_published = False
        self.ore_requested = False
        self.ore_done = False
        self.finished = False
        self.package_taken = False
        self.ready_sent = False
        self.nav_done = False
        self.nav_done_time = 0.0
        self.next_plan_time = 0.0
        self.next_ore_time = 0.0
        self.next_wait_log = 0.0
        self.run_start_time = self.now_s()
        self.publish_ready(False)
        self.get_logger().info("Planner reset for a new run")


    # ------------------------- callbacks -------------------------

    def mapcb(self, data):
        '''Callback for /map. A changed map, or any map after a finished run, starts a new run.'''
        raw = np.array(data.data, dtype=np.int8)
        same_map = self.map_bytes is not None and raw.tobytes() == self.map_bytes
        if self.grid is not None and same_map and not self.finished:
            return                                  # same map re-published mid-run: ignore

        info = data.info
        width = info.width
        height = info.height
        resolution = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        flat = raw.astype(np.int16).reshape(height, width)
        occ = flat >= occ_threshold

        self.map_bytes = raw.tobytes()
        self.grid = {
            'width': width,
            'height': height,
            'resolution': resolution,
            'origin_x': origin_x,
            'origin_y': origin_y,
            'occ': occ
        }

        self.get_logger().info(
            f"Map received: {width}x{height} cells, resolution {resolution:.3f} m, "
            f"origin ({origin_x:.2f}, {origin_y:.2f}), rock cells {int(occ.sum())}")

        self.coarse_resolution = plan_factor * resolution
        self.coarse = self.build_coarse()
        self.get_logger().info(
            f"Planning grid: {self.coarse.shape[1]}x{self.coarse.shape[0]} cells "
            f"@ {self.coarse_resolution:.3f} m")

        self.reset_run()


    def odomcb(self, data):
        '''Callback for /odom: stores the pose.'''
        p = data.pose.pose.position
        q = data.pose.pose.orientation

        self.x = p.x
        self.y = p.y
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.odom = (self.x, self.y, self.yaw)


    def takencb(self, msg):
        '''Callback for /handover/package_taken (arm -> eBot).'''
        if msg.data and not self.package_taken:
            self.get_logger().info("Arm reports: package taken, eBot may leave")
        self.package_taken = bool(msg.data)


    def nav_donecb(self, msg):
        '''Callback for /ebot_nav/step_done (nav -> planner).'''
        if msg.data and not self.nav_done:
            self.nav_done_time = self.now_s()
        self.nav_done = bool(msg.data)


    # ------------------------- helpers -------------------------

    @staticmethod
    def wrap_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

    def at_pose(self, pose, allow_flip=False):
        '''True if within pos_tol and yaw_tol of pose; allow_flip also accepts yaw + pi.'''
        if math.hypot(self.x - pose[0], self.y - pose[1]) > pos_tol:
            return False
        err = abs(self.wrap_angle(self.yaw - pose[2]))
        if allow_flip:
            err = min(err, abs(self.wrap_angle(self.yaw - pose[2] - math.pi)))
        return err <= yaw_tol

    def build_coarse(self):
        '''Shrink the fine map: a planning cell is a rock if ANY fine cell in it is a rock.'''
        occ = self.grid['occ']
        f = plan_factor
        h, w = occ.shape
        padded = np.pad(occ, ((0, (-h) % f), (0, (-w) % f)), constant_values=False)
        ph, pw = padded.shape
        return padded.reshape(ph // f, f, pw // f, f).any(axis=(1, 3))

    def world_to_cell(self, x, y):
        col = int(math.floor((x - self.grid['origin_x']) / self.coarse_resolution + 1e-6))
        row = int(math.floor((y - self.grid['origin_y']) / self.coarse_resolution + 1e-6))
        return row, col

    def cell_to_world(self, row, col):
        x = self.grid['origin_x'] + (col + 0.5) * self.coarse_resolution
        y = self.grid['origin_y'] + (row + 0.5) * self.coarse_resolution
        return x, y

    def disc_offsets(self, radius):
        r = int(math.ceil(radius / self.coarse_resolution))
        offsets = []
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if math.hypot(dr, dc) * self.coarse_resolution <= radius:
                    offsets.append((dr, dc))
        return offsets

    def inflate(self, radius, keep_free=()):
        '''Grow every rock by 'radius' metres. True = blocked. keep_free: points forced free.'''
        rows, cols = self.coarse.shape
        blocked = np.zeros((rows, cols), dtype=bool)

        for dr, dc in self.disc_offsets(radius):
            src_r0, src_r1 = max(0, -dr), min(rows, rows - dr)
            src_c0, src_c1 = max(0, -dc), min(cols, cols - dc)
            blocked[src_r0 + dr:src_r1 + dr, src_c0 + dc:src_c1 + dc] |= \
                self.coarse[src_r0:src_r1, src_c0:src_c1]

        for x, y in keep_free:
            r, c = self.world_to_cell(x, y)
            for dr, dc in self.disc_offsets(clear_radius):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    blocked[rr, cc] = False
        return blocked

    def astar(self, blocked, near, start, goal):
        '''8-connected A* on the planning grid. Returns a list of (row, col), or None.'''
        rows, cols = blocked.shape
        root2 = math.sqrt(2.0)
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, root2), (-1, 1, root2), (1, -1, root2), (1, 1, root2)]

        def heuristic(cell):
            dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
            return (dr + dc) + (root2 - 2.0) * min(dr, dc)

        g_cost = {start: 0.0}
        parent = {}
        closed = set()
        heap = [(heuristic(start), start)]

        while heap:
            _, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            if cur == goal:
                path = [cur]
                while cur in parent:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path
            closed.add(cur)

            r, c = cur
            for dr, dc, step in moves:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if blocked[nr, nc]:
                    continue
                if dr != 0 and dc != 0 and (blocked[r + dr, c] or blocked[r, c + dc]):
                    continue
                if near[nr, nc]:
                    step *= (1.0 + near_penalty)
                new_g = g_cost[cur] + step
                nxt = (nr, nc)
                if new_g < g_cost.get(nxt, float('inf')):
                    g_cost[nxt] = new_g
                    parent[nxt] = cur
                    heapq.heappush(heap, (new_g + heuristic(nxt), nxt))
        return None

    def line_free(self, blocked, a, b):
        (r0, c0), (r1, c1) = a, b
        dist = math.hypot(r1 - r0, c1 - c0)
        n = max(1, int(math.ceil(dist * 2)))
        for k in range(n + 1):
            t = k / n
            r = int(round(r0 + (r1 - r0) * t))
            c = int(round(c0 + (c1 - c0) * t))
            if blocked[r, c]:
                return False
        return True

    def smooth(self, cells, safe):
        '''Line-of-sight shortcutting against the 'safe' map (bigger clearance than A*).'''
        result = [cells[0]]
        i = 0
        last = len(cells) - 1
        while i < last:
            j = last
            while j > i + 1 and not self.line_free(safe, cells[i], cells[j]):
                j -= 1
            result.append(cells[j])
            i = j
        return result

    def resample(self, pts, spacing):
        out = [pts[0]]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            d = math.hypot(x1 - x0, y1 - y0)
            if d < 1e-9:
                continue
            n = max(1, int(math.ceil(d / spacing)))
            for k in range(1, n + 1):
                t = k / n
                out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
        return out


    def plan_path(self, start, end):
        '''Plans a path from 'start' to 'end' clear of every rock; None if no path found.'''
        points = None

        keep = [(start[0], start[1]), (end[0], end[1])]
        s_cell = self.world_to_cell(start[0], start[1])
        g_cell = self.world_to_cell(end[0], end[1])

        for margin in margin_steps:
            blocked = self.inflate(robot_radius + margin, keep_free=keep)
            near = self.inflate(robot_radius + margin + near_band, keep_free=keep)
            safe = self.inflate(robot_radius + margin + smooth_extra, keep_free=keep)
            cells = self.astar(blocked, near, s_cell, g_cell)
            if cells is None:
                self.get_logger().warn(f"No path with margin {margin:.2f} m, trying a smaller one")
                continue

            corners = self.smooth(cells, safe)
            pts = [self.cell_to_world(r, c) for r, c in corners]
            pts[0] = (start[0], start[1])
            pts[-1] = (end[0], end[1])
            points = self.resample(pts, path_spacing)
            length = sum(math.hypot(points[i + 1][0] - points[i][0],
                                    points[i + 1][1] - points[i][1])
                         for i in range(len(points) - 1))
            self.get_logger().info(
                f"Path found with margin {margin:.2f} m: {len(points)} points, {length:.2f} m")
            break

        if points is None:
            self.get_logger().error("No path found at any margin")

        return points


    def publish_path(self, points, start, end):
        '''Publishes 'points' on /ebot_path as a nav_msgs/Path.'''
        path = Path()
        path.header.frame_id = map_frame
        path.header.stamp = self.get_clock().now().to_msg()

        n = len(points)
        for i, (x, y) in enumerate(points):
            if i == 0:
                yaw = start[2]
            elif i == n - 1:
                yaw = end[2]
            else:
                nx, ny = points[i + 1]
                yaw = math.atan2(ny - y, nx - x)

            pose = PoseStamped()
            pose.header.frame_id = map_frame
            pose.header.stamp = path.header.stamp
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(pose)

        self.nav_done = False               # the nav node will report True when this leg ends
        self.path_pub.publish(path)
        self.get_logger().info(f"Published path with {n} points")


    def publish_ready(self, value):
        '''Publishes /handover/ebot_ready (eBot -> arm).'''
        msg = Bool()
        msg.data = bool(value)
        self.ready_pub.publish(msg)


    def request_ore_package(self):
        '''Calls /spawn_ore_package once the eBot stands still on the ore drop pose.'''
        if self.ore_requested or self.ore_done:
            return

        now = self.now_s()
        if now < self.next_ore_time:
            return

        if not self.spawn_client.service_is_ready():
            self.get_logger().warn("Spawn service not ready", throttle_duration_sec=1.0)
            return

        self.ore_requested = True
        self.next_ore_time = now + ore_retry_s
        future = self.spawn_client.call_async(Trigger.Request())
        future.add_done_callback(self.ore_reply_cb)
        self.get_logger().info("Ore package requested")


    def ore_reply_cb(self, future):
        '''Called when /spawn_ore_package replies.'''
        try:
            response = future.result()
            self.get_logger().info(
                f"Ore package reply: success={response.success}, message='{response.message}'")
            self.ore_done = bool(response.success)
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            self.ore_done = False
        finally:
            self.ore_requested = False


    def process_planning(self):
        '''Timer: plan, publish, request and hand over, step by step.'''
        try:
            now = self.now_s()

            # backup run detection: simulation time jumped back => a new world started
            if now < self.last_now - 1.0:
                self.get_logger().info("Simulation clock went backwards: new run")
                self.reset_run()
            self.last_now = now

            if self.grid is None or self.odom is None or self.finished:
                return

            start, end = steps[self.step]
            start_flip = self.step > 0
            end_flip = self.step < 2

            # 1) plan and publish at the start of the step
            if not self.path_published:
                if self.step == LEAVE_STEP and not self.package_taken:
                    if now >= self.next_wait_log:
                        self.next_wait_log = now + wait_log_s
                        self.get_logger().info("Waiting for the arm to take the package")
                    return
                if self.step == 0 and now < self.run_start_time + start_delay:
                    return
                if not self.at_pose(start, allow_flip=start_flip):
                    self.get_logger().info(
                        f"Waiting at start: pos=({self.x:.2f},{self.y:.2f}) yaw={self.yaw:.2f}",
                        throttle_duration_sec=3.0)
                    return
                if now < self.next_plan_time:
                    return
                points = self.plan_path(start, end)
                if points is None:
                    self.next_plan_time = now + plan_retry_s
                    self.get_logger().error(
                        f"Step {self.step + 1}: no path, retrying in {plan_retry_s:.0f} s")
                    return
                self.publish_path(points, start, end)
                self.path_published = True
                return

            # 2) wait until the navigation node reports the step finished
            if not (self.nav_done and now - self.nav_done_time >= end_settle):
                return
            if not self.at_pose(end, allow_flip=end_flip):
                self.get_logger().warn(
                    f"Nav finished but pose ({self.x:.2f},{self.y:.2f},{self.yaw:.2f}) "
                    f"is outside the end tolerance", throttle_duration_sec=2.0)
                return

            # 3) after step 1, the ore package must be dropped before moving on
            if self.step == 0 and not self.ore_done:
                self.request_ore_package()
                return

            # 4) beside the arm: tell the arm
            if self.step == ARM_STEP and not self.ready_sent:
                self.publish_ready(True)
                self.ready_sent = True
                self.get_logger().info("eBot beside the arm: /handover/ebot_ready = True")

            # 5) step complete
            self.get_logger().info(f"Step {self.step + 1} complete")
            self.step += 1
            self.path_published = False
            if self.step >= len(steps):
                self.finished = True
                self.get_logger().info("Lap complete")

        except Exception as e:
            self.get_logger().error(f"process_planning error: {e}", throttle_duration_sec=1.0)


##################### FUNCTION DEFINITION #######################

def main():
    rclpy.init(args=sys.argv)

    node = rclpy.create_node('ebot_planner_process')
    node.get_logger().info('Node created: eBot planner process')

    ebot_planner_class = ebot_planner()

    try:
        rclpy.spin(ebot_planner_class)
    except KeyboardInterrupt:
        pass
    finally:
        ebot_planner_class.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()