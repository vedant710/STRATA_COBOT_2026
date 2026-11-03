#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 2B of StrataCobot (SC) Theme (eYRC 2026-27).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          [ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:		    task2B_planning.py
# Functions:
#			        [ mapcb, odomcb, wrap_angle, is_still, at_pose, build_coarse,
#                     world_to_cell, cell_to_world, disc_offsets, inflate, astar,
#                     line_free, smooth, resample, plan_path, publish_path,
#                     request_ore_package, ore_reply_cb, process_planning, main ]
# Nodes:		    ebot_planner_node
#			        Publishing Topics  - [ /ebot_path ]
#                   Subscribing Topics - [ /map, /odom ]
#                   Service Clients    - [ /spawn_ore_package ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
import heapq
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_srvs.srv import Trigger


##################### TASK CONSTANTS #######################

# The map is published ONCE, as the world loads, and held for late subscribers.
map_topic = '/map'

# Your follower reads the path from here, with the same held-message QoS.
path_topic = '/ebot_path'

odom_topic = '/odom'
spawn_service = '/spawn_ore_package'

# Every pose in the path, and the path itself, is in this frame.
map_frame = 'map'

# The three points of the lap: (x, y) in metres and yaw in radians, in the map frame.
# At the ore drop pose and the arm pose either yaw, 0.0 or 3.14, is valid.
# The arm pose faces back across the arena because step 2 arrives travelling left.
home_pose = (0.0, 0.0, 0.0)
ore_drop_pose = (6.0, 1.9, 0.0)
arm_pose = (0.0, 1.9, math.pi)

# The lap, as (start, end) of each step, in the order they are driven.
steps = [
    (home_pose, ore_drop_pose),     # step 1: then request the ore package
    (ore_drop_pose, arm_pose),      # step 2
    (arm_pose, home_pose),          # step 3
]

# ---- map reading ----
occ_threshold = 50        # cells with occupancy >= this are rocks
plan_factor = 5           # fine cells per planning cell (0.005 m -> 0.025 m); use 10 if too slow

# ---- standing still and end-of-step tolerances ----
still_lin = 0.02          # m/s: below this the base counts as not moving
still_ang = 0.02          # rad/s
pos_tol = 0.30            # m: end-of-step position tolerance
yaw_tol = 0.15            # rad: end-of-step yaw tolerance
settle_time = 1.0         # s: must stay still this long

# ---- planning ----
robot_radius = 0.40                                   # m: circle enclosing the 0.62 x 0.49 m chassis
margin_steps = (0.08, 0.05, 0.03, 0.01, 0.0)          # m: extra room beyond robot_radius, widest first
clear_radius = 0.30                                   # m: disc around start and goal forced free
near_band = 0.15                                      # m: band beyond the margin that costs more to cross
near_penalty = 1.0                                # extra cost multiplier inside that band
path_spacing = 0.20                                 # m: distance between published path points

# ---- retry timing ----
plan_retry_s = 2.0        # s: wait after a failed plan
ore_retry_s = 1.0         # s: wait between ore service calls


##################### CLASS DEFINITION #######################

class ebot_planner(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to read the map, plan a path for each
                    step of the lap, publish it on /ebot_path, and request the ore package
                    at the end of step 1.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ebot_planner
        '''

        # use_sim_time is set here, not on the command line, so this node runs on the
        # simulation clock however it is started.
        super().__init__(                                                               # registering node
            'ebot_planner_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        # A held message: kept by the publisher and handed to anyone who subscribes later.
        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        ############ Topic SUBSCRIPTIONS ############

        self.map_sub = self.create_subscription(OccupancyGrid, map_topic, self.mapcb, latched)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)

        ############ Topic PUBLISHERS ############

        self.path_pub = self.create_publisher(Path, path_topic, latched)                # your follower drives what you publish here

        ############ Service CLIENTS ############

        self.spawn_client = self.create_client(Trigger, spawn_service)                  # drops the ore package onto the eBot

        ############ Constructor VARIABLES/OBJECTS ############

        plan_rate = 0.1                                                                 # rate of time to run one planning cycle (seconds)
        self.timer = self.create_timer(plan_rate, self.process_planning)                # creating a timer based function which gets called on every 0.1 seconds (as defined by 'plan_rate' variable)

        self.grid = None                                                                # the map (from mapcb())
        self.odom = None                                                                # where the base is (from odomcb())
        self.step = 0                                                                   # which step of the lap is next (index into 'steps')

        ############ ADD YOUR CODE HERE ############

        # odometry variables
        self.x = 0.0                    # base x in the map frame (m)
        self.y = 0.0                    # base y (m)
        self.yaw = 0.0                  # heading (rad)
        self.linear_vel = 0.0           # forward speed (m/s)
        self.angular_vel = 0.0          # turn rate (rad/s)
        self.still_since = None         # sim time (s) when the base last became still

        # state machine variables
        self.path_published = False     # has the current step's path been published?
        self.ore_requested = False      # is a service call in flight?
        self.ore_done = False           # did the service reply success: true?
        self.finished = False           # True once step 3 is complete
        self.next_plan_time = 0.0       # earliest sim time for the next planning attempt
        self.next_ore_time = 0.0        # earliest sim time for the next ore call

        # planning grid variables
        self.coarse = None              # planning grid: True = rock (not inflated)
        self.coarse_resolution = None   # metres per planning cell

        ############################################


    def mapcb(self, data):
        '''
        Description:    Callback function for the map topic.
                        Use this function to receive the occupancy grid of the arena.

        Args:
            data (OccupancyGrid):    The arena, one value per cell

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # the map is latched and arrives once; keep the first copy, ignore repeats
        if self.grid is not None:
            return

        info = data.info

        width = info.width
        height = info.height
        resolution = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        # flat list -> 2D array, so occ[row, col] is True where there is a rock
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

        rock_cells = int(occ.sum())
        self.get_logger().info(
            f"Map received: {width}x{height} cells, resolution {resolution:.3f} m, "
            f"origin ({origin_x:.2f}, {origin_y:.2f}), rock cells {rock_cells}")
        self.get_logger().info(
            f"Map covers x: {origin_x:.2f} to {origin_x + width * resolution:.2f} m, "
            f"y: {origin_y:.2f} to {origin_y + height * resolution:.2f} m")

        self.coarse_resolution = plan_factor * resolution
        self.coarse = self.build_coarse()
        self.get_logger().info(
            f"Planning grid: {self.coarse.shape[1]}x{self.coarse.shape[0]} cells "
            f"@ {self.coarse_resolution:.3f} m, rock cells {int(self.coarse.sum())}")

        ############################################


    def odomcb(self, data):
        '''
        Description:    Callback function for the odometry topic.
                        Use this function to receive where the base currently is.

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

        # track how long the base has been continuously still
        now = self.get_clock().now().nanoseconds * 1e-9
        moving = abs(self.linear_vel) >= still_lin or abs(self.angular_vel) >= still_ang
        if moving:
            self.still_since = None
        elif self.still_since is None:
            self.still_since = now

        self.odom = (self.x, self.y, self.yaw)

        ############################################


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
        '''
        Grow every rock by 'radius' metres. Returns a boolean grid, True = blocked.
        keep_free: list of (x, y) points to force free (start and goal).
        '''
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
        '''
        8-connected A* on the planning grid.
        blocked: bool grid, True = forbidden.  near: bool grid, True = allowed but costly.
        start, goal: (row, col).  Returns a list of (row, col), or None.
        '''
        rows, cols = blocked.shape
        root2 = math.sqrt(2.0)
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, root2), (-1, 1, root2), (1, -1, root2), (1, 1, root2)]

        def heuristic(cell):
            dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
            return (dr + dc) + (root2 - 2.0) * min(dr, dc)      # octile distance

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
                    continue                                    # no corner cutting
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
        n = max(1, int(math.ceil(dist * 2)))               # a sample every half cell
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

        # try the widest safety margin first and squeeze only when no path exists
        for margin in margin_steps:
            blocked = self.inflate(robot_radius + margin, keep_free=keep)
            near = self.inflate(robot_radius + margin + near_band, keep_free=keep)
            cells = self.astar(blocked, near, s_cell, g_cell)
            if cells is None:
                self.get_logger().warn(f"No path with margin {margin:.2f} m, trying a smaller one")
                continue

            corners = self.smooth(cells, blocked)
            pts = [self.cell_to_world(r, c) for r, c in corners]
            pts[0] = (start[0], start[1])                   # exact endpoints
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
                yaw = start[2]                                  # exact start yaw
            elif i == n - 1:
                yaw = end[2]                                    # exact end yaw
            else:
                nx, ny = points[i + 1]
                yaw = math.atan2(ny - y, nx - x)                # face along the path

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
        Description:    Called when /spawn_ore_package replies. Marks the package as
                        dropped only if the service reported success.

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
        Description:    Timer function used to plan, publish and request, step by step.

        Args:
        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:
            if self.grid is None or self.odom is None or self.finished:
                return

            start, end = steps[self.step]
            start_flip = self.step > 0          # ore drop and arm poses accept yaw 0 or 3.14
            end_flip = self.step < 2            # home must end at yaw 0.0

            # 1) plan and publish while standing still at the start
            if not self.path_published:
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

            # 4) step complete
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

    rclpy.init(args=sys.argv)                                           # initialisation

    node = rclpy.create_node('ebot_planner_process')                    # creating ROS node

    node.get_logger().info('Node created: eBot planner process')        # logging information

    ebot_planner_class = ebot_planner()                                 # creating a new object for class 'ebot_planner'

    rclpy.spin(ebot_planner_class)                                      # spining on the object to make it alive in ROS 2 DDS

    ebot_planner_class.destroy_node()                                   # destroy node after spin ends

    rclpy.shutdown()                                                    # shutdown process


if __name__ == '__main__':

    main()