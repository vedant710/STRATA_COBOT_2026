#!/usr/bin/env python3



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

# handover topics (your own interfaces)
ready_topic = '/handover/ebot_ready'
taken_topic = '/handover/package_taken'

map_frame = 'map'

# The three points of the lap: (x, y, yaw) in the map frame.
home_pose = (0.0, 0.0, 0.0)
ore_drop_pose = (6.0, 1.9, 0.0)

# TODO (MEASURE THIS): the stop pose beside the arm. It is NOT given by the task.
# Placeholder = the Task 2B arm pose. Find the arm base in the map frame, then pick a
# pose where the package on the deck is inside the arm's comfortable reach (~0.4-0.7 m).
# Yaw 0.0 or 3.14 are both accepted for the leg to the arm.
arm_pose = (0.0, 1.9, math.pi)

steps = [
    (home_pose, ore_drop_pose),     # step 1: then request the ore package
    (ore_drop_pose, arm_pose),      # step 2: then tell the arm the eBot is ready
    (arm_pose, home_pose),          # step 3: only after the arm has taken the package
]
ARM_STEP = 1                        # index of the step that ends beside the arm
LEAVE_STEP = 2                      # index of the step that waits for package_taken

# ---- map reading ----
occ_threshold = 50
plan_factor = 5

# ---- standing still and end-of-step tolerances ----
still_lin = 0.02
still_ang = 0.02
pos_tol = 0.30
yaw_tol = 0.15
settle_time = 1.0

# ---- planning ----
robot_radius = 0.40
margin_steps = (0.08, 0.05, 0.03, 0.01, 0.0)
clear_radius = 0.30
near_band = 0.15
near_penalty = 1.0
path_spacing = 0.20

# ---- retry timing ----
plan_retry_s = 2.0
ore_retry_s = 1.0
wait_log_s = 5.0


##################### CLASS DEFINITION #######################

class ebot_planner(Node):
    '''
    ___CLASS___

    Description:    Plans each leg, publishes it on /ebot_path, requests the ore package
                    at the ore drop pose, tells the arm when the eBot has arrived, and
                    holds the eBot beside the arm until the arm has taken the package.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ebot_planner
        '''

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

        ############ Topic PUBLISHERS ############

        self.path_pub = self.create_publisher(Path, path_topic, latched)
        self.ready_pub = self.create_publisher(Bool, ready_topic, latched)

        ############ Service CLIENTS ############

        self.spawn_client = self.create_client(Trigger, spawn_service)

        ############ Constructor VARIABLES/OBJECTS ############

        plan_rate = 0.1
        self.timer = self.create_timer(plan_rate, self.process_planning)

        self.grid = None
        self.odom = None
        self.step = 0

        ############ ADD YOUR CODE HERE ############

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.still_since = None

        self.path_published = False
        self.ore_requested = False
        self.ore_done = False
        self.finished = False
        self.next_plan_time = 0.0
        self.next_ore_time = 0.0
        self.next_wait_log = 0.0

        # handover state (reset on every start of this node)
        self.package_taken = False
        self.ready_sent = False

        self.coarse = None
        self.coarse_resolution = None

        self.publish_ready(False)               # arm must not act on a stale True

        ############################################


    def mapcb(self, data):
        '''
        Description:    Callback function for the map topic.

        Args:
            data (OccupancyGrid):    The arena, one value per cell

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        if self.grid is not None:
            return

        info = data.info
        width = info.width
        height = info.height
        resolution = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        flat = np.array(data.data, dtype=np.int16).reshape(height, width)
        occ = flat >= occ_threshold

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

        ############################################


    def odomcb(self, data):
        '''
        Description:    Callback function for the odometry topic.

        Args:
            data (Odometry):    Pose and velocity of the base

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        p = data.pose.pose.position
        q = data.pose.pose.orientation

        self.x = p.x
        self.y = p.y
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

        self.linear_vel = data.twist.twist.linear.x
        self.angular_vel = data.twist.twist.angular.z

        now = self.get_clock().now().nanoseconds * 1e-9
        moving = abs(self.linear_vel) >= still_lin or abs(self.angular_vel) >= still_ang
        if moving:
            self.still_since = None
        elif self.still_since is None:
            self.still_since = now

        self.odom = (self.x, self.y, self.yaw)

        ############################################


    def takencb(self, msg):
        '''
        Description:    Callback for /handover/package_taken (arm -> eBot).

        Args:
            msg (Bool):    True once the package is lifted clear of the deck

        Returns:
        '''

        if msg.data and not self.package_taken:
            self.get_logger().info("Arm reports: package taken, eBot may leave")
        self.package_taken = bool(msg.data)


    @staticmethod
    def wrap_angle(a):
        '''wrap an angle to [-pi, pi]'''
        return math.atan2(math.sin(a), math.cos(a))

    def is_still(self):
        '''True if the base has been still for settle_time seconds'''
        if self.still_since is None:
            return False
        now = self.get_clock().now().nanoseconds * 1e-9
        return (now - self.still_since) >= settle_time

    def at_pose(self, pose, allow_flip=False):
        '''True if within pos_tol and yaw_tol of pose (x, y, yaw).
        allow_flip also accepts yaw + pi (0.0 and +-3.14 are both valid).'''
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
        '''world (x, y) in metres -> planning cell (row, col)'''
        col = int(math.floor((x - self.grid['origin_x']) / self.coarse_resolution + 1e-6))
        row = int(math.floor((y - self.grid['origin_y']) / self.coarse_resolution + 1e-6))
        return row, col

    def cell_to_world(self, row, col):
        '''planning cell (row, col) -> world (x, y) of the cell CENTRE'''
        x = self.grid['origin_x'] + (col + 0.5) * self.coarse_resolution
        y = self.grid['origin_y'] + (row + 0.5) * self.coarse_resolution
        return x, y

    def disc_offsets(self, radius):
        '''(drow, dcol) offsets of every planning cell within 'radius' metres of a centre cell.'''
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
        '''True if the straight segment between cells a and b stays in free cells.'''
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

    def smooth(self, cells, blocked):
        '''Line-of-sight shortcutting: keep only the cells where the path has to turn.'''
        result = [cells[0]]
        i = 0
        last = len(cells) - 1
        while i < last:
            j = last
            while j > i + 1 and not self.line_free(blocked, cells[i], cells[j]):
                j -= 1
            result.append(cells[j])
            i = j
        return result

    def resample(self, pts, spacing):
        '''Put a point every 'spacing' metres along the polyline, keeping every corner.'''
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
        '''
        Description:    Plans a path from 'start' to 'end' that stays clear of every rock.

        Args:
            start (tuple):    (x, y, yaw) where the step begins, in the map frame
            end (tuple):      (x, y, yaw) where the step ends, in the map frame

        Returns:
            list:             [(x, y), ...] from 'start' to 'end', or None if no path was found
        '''

        points = None

        ############ ADD YOUR CODE HERE ############

        keep = [(start[0], start[1]), (end[0], end[1])]
        s_cell = self.world_to_cell(start[0], start[1])
        g_cell = self.world_to_cell(end[0], end[1])

        for margin in margin_steps:
            blocked = self.inflate(robot_radius + margin, keep_free=keep)
            near = self.inflate(robot_radius + margin + near_band, keep_free=keep)
            cells = self.astar(blocked, near, s_cell, g_cell)
            if cells is None:
                self.get_logger().warn(f"No path with margin {margin:.2f} m, trying a smaller one")
                continue

            corners = self.smooth(cells, blocked)
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

        ############################################

        return points


    def publish_path(self, points, start, end):
        '''
        Description:    Publishes 'points' on /ebot_path as a nav_msgs/Path.

        Args:
            points (list):    [(x, y), ...] as returned by plan_path()
            start (tuple):    (x, y, yaw) where the step begins
            end (tuple):      (x, y, yaw) where the step ends

        Returns:
        '''

        path = Path()
        path.header.frame_id = map_frame
        path.header.stamp = self.get_clock().now().to_msg()

        ############ ADD YOUR CODE HERE ############

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

        self.path_pub.publish(path)
        self.get_logger().info(f"Published path with {n} points")

        ############################################


    def publish_ready(self, value):
        '''
        Description:    Publishes /handover/ebot_ready (eBot -> arm).

        Args:
            value (bool):    True when the eBot stands still beside the arm with the package

        Returns:
        '''

        msg = Bool()
        msg.data = bool(value)
        self.ready_pub.publish(msg)


    def request_ore_package(self):
        '''
        Description:    Calls /spawn_ore_package once the eBot stands still on the ore drop pose.

        Args:
        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        if self.ore_requested or self.ore_done:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
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

        ############################################


    def ore_reply_cb(self, future):
        '''
        Description:    Called when /spawn_ore_package replies.

        Args:
            future (Future):    The finished service call

        Returns:
        '''

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
        '''
        Description:    Timer function used to plan, publish, request and hand over,
                        step by step.

        Args:
        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:
            if self.grid is None or self.odom is None or self.finished:
                return

            start, end = steps[self.step]
            start_flip = self.step > 0
            end_flip = self.step < 2

            # 1) plan and publish while standing still at the start
            if not self.path_published:
                # leg 3: the eBot stays beside the arm until the package is off its deck
                if self.step == LEAVE_STEP and not self.package_taken:
                    now = self.get_clock().now().nanoseconds * 1e-9
                    if now >= self.next_wait_log:
                        self.next_wait_log = now + wait_log_s
                        self.get_logger().info("Waiting for the arm to take the package")
                    return
                if not (self.is_still() and self.at_pose(start, allow_flip=start_flip)):
                    return
                now = self.get_clock().now().nanoseconds * 1e-9
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

            # 2) wait until the follower has finished the step
            if not (self.is_still() and self.at_pose(end, allow_flip=end_flip)):
                return

            # 3) after step 1, the ore package must be dropped before moving on
            if self.step == 0 and not self.ore_done:
                self.request_ore_package()
                return

            # 4) beside the arm: tell the arm, then continue (step 3 waits for package_taken)
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

        ############################################


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    ebot_planner class to perform its task
    '''

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