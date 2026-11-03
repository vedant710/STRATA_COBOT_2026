#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1C of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:		    task1c_script.py
# Functions:
#			        [ pathcb, mapcb, odomcb, scancb, process_navigation, _navigate,
#                     _occupied, _lidar_free, _map_free, _wrap, main ]
# Nodes:
#			        Publishing Topics  - [ /cmd_vel ]
#                   Subscribing Topics - [ /ebot_path, /map, /odom, /scan, /tf, /tf_static ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                       HistoryPolicy, qos_profile_sensor_data)
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener


##################### TASK CONSTANTS #######################

# The route is published here ONCE, before your node starts, and held for late
# subscribers. An ordinary subscription waits forever for a message already sent.
path_topic = '/ebot_path'
map_topic = '/map'

odom_topic = '/odom'
scan_topic = '/scan'
cmd_topic = '/cmd_vel'

# What /cmd_vel accepts. The base clips anything beyond these, and clipping one of the
# two changes their ratio - which is the arc the base actually drives.
max_linear_mps = 0.5
max_angular_rps = 1.0

# ---- controller tuning ----
HALF_WIDTH = 0.33        # half the swept width of the eBot (check the URDF: widest part is not the chassis)
LOOKAHEAD = 1.2          # how far ahead (m) we look for obstacles
V_CRUISE = 0.30          # normal driving speed (m/s)
KP_HEADING = 1.5         # heading P gain
MIN_TURN = 0.30          # minimum turn rate when rotating on the spot (skid steer needs a push)
WP_TOL = 0.35            # intermediate waypoint accepted inside this distance (scoring uses 0.5)
LAST_TOL = 0.20          # final waypoint tolerance
OCC_THRESH = 50          # occupancy value (0-100) treated as an obstacle


##################### CLASS DEFINITION #######################

class ebot_nav(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to drive the eBot along the route
                    published on /ebot_path, through its waypoints, in order.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ebot_nav
        '''

        # use_sim_time is set here, not on the command line, so this node runs on the
        # simulation clock however it is started.
        super().__init__(                                                               # registering node
            'ebot_nav_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        ############ Topic SUBSCRIPTIONS ############

        # latched QoS so the held route and map are delivered
        latched = QoSProfile(depth=1,
                             history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_sub = self.create_subscription(Path, path_topic, self.pathcb, latched)
        self.map_sub = self.create_subscription(OccupancyGrid, map_topic, self.mapcb, latched)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scancb, qos_profile_sensor_data)

        ############ Topic PUBLISHERS ############

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)                      # the eBot drives on what you publish here

        ############ Constructor VARIABLES/OBJECTS ############

        control_rate = 0.05                                                             # rate of time to run one control cycle (seconds)
        self.timer = self.create_timer(control_rate, self.process_navigation)           # creating a timer based function which gets called on every 0.05 seconds (as defined by 'control_rate' variable)

        self.route = None                                                               # the route to drive (from pathcb())
        self.odom = None                                                                # where the base is (from odomcb())
        self.scan = None                                                                # what the lidar sees (from scancb())

        ############ ADD YOUR CODE HERE ############

        self.wp_idx = 0              # index of the waypoint we are heading to
        self.done = False            # True once the route is completed
        self.map = None              # occupancy grid info (from mapcb())
        self.prev_theta = 0.0        # last chosen steering direction (hysteresis)

        # lidar mounting: looked up from /tf, never typed in
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.base_frame = 'base_link'
        self.lidar_tf = None         # (x, y, yaw) of the lidar in the base frame
        ############################################


    def pathcb(self, data):
        '''
        Description:    Callback function for the route topic.
                        Use this function to receive the waypoints the eBot has to drive.

        Args:
            data (Path):    The route, as a sequence of poses

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        if self.route is not None:          # take the route once only
            return
        self.route = [(p.pose.position.x, p.pose.position.y) for p in data.poses]

        self.get_logger().info(f"Received route with {len(self.route)} waypoints, frame {data.header.frame_id}")
        ############################################


    def mapcb(self, data):
        '''
        Description:    Callback function for the occupancy grid (rock layout).

        Args:
            data (OccupancyGrid):    The rock map

        Returns:
        '''

        if self.map is not None:
            return
        self.map = {
            'res': data.info.resolution,
            'w': data.info.width,
            'h': data.info.height,
            'ox': data.info.origin.position.x,
            'oy': data.info.origin.position.y,
            'data': data.data,
        }
        self.get_logger().info(f"Received map {data.info.width}x{data.info.height} @ {data.info.resolution:.3f} m")


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
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.odom = (p.x, p.y, yaw)

        if data.child_frame_id:
            self.base_frame = data.child_frame_id

        self.get_logger().info(f"Received odometry: x={p.x:.2f}, y={p.y:.2f}, yaw={math.degrees(yaw):.0f}",
                               throttle_duration_sec=1.0)
        ############################################


    def scancb(self, data):
        '''
        Description:    Callback function for the lidar topic.
                        Use this function to receive what the lidar currently sees.

        Args:
            data (LaserScan):    One lidar sweep

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # where the lidar sits on the base (static, so look it up until found, then keep it)
        if self.lidar_tf is None:
            try:
                t = self.tf_buffer.lookup_transform(self.base_frame, data.header.frame_id, Time())
                tr = t.transform.translation
                q = t.transform.rotation
                lyaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                self.lidar_tf = (tr.x, tr.y, lyaw)
                self.get_logger().info(f"Lidar offset from {self.base_frame}: x={tr.x:.3f} y={tr.y:.3f}")
            except Exception:
                self.lidar_tf = None
        lx, ly, lyaw = self.lidar_tf if self.lidar_tf is not None else (0.0, 0.0, 0.0)

        # valid beams only, converted to x, y in the BASE frame
        points = []
        for i, r in enumerate(data.ranges):
            if math.isfinite(r) and data.range_min <= r <= data.range_max:
                a = data.angle_min + i * data.angle_increment + lyaw
                points.append((lx + r * math.cos(a), ly + r * math.sin(a)))
        self.scan = points

        if points:
            self.get_logger().info(
                f"valid beams= {len(points)} nearest= {min(math.hypot(px, py) for px, py in points):.2f} m",
                throttle_duration_sec=1.0)
        ############################################


    # ------------------------- helpers -------------------------

    @staticmethod
    def _wrap(e):
        '''wrap an angle to [-pi, pi]'''
        return math.atan2(math.sin(e), math.cos(e))

    def _occupied(self, wx, wy):
        '''True if the world point (wx, wy) lies in an occupied cell of /map'''
        m = self.map
        if m is None:
            return False
        c = int((wx - m['ox']) / m['res'])
        r = int((wy - m['oy']) / m['res'])
        if c < 0 or r < 0 or c >= m['w'] or r >= m['h']:
            return False
        return m['data'][r * m['w'] + c] >= OCC_THRESH

    def _lidar_free(self, theta):
        '''free distance along base-frame direction theta, from the lidar'''
        ct, st = math.cos(theta), math.sin(theta)
        free = LOOKAHEAD
        for px, py in self.scan:
            along = px * ct + py * st
            if 0.0 < along < free and abs(-px * st + py * ct) < HALF_WIDTH:
                free = along
        return free

    def _map_free(self, x, y, yaw, theta):
        '''free distance along direction theta, from the /map rock layout'''
        if self.map is None:
            return LOOKAHEAD
        a = yaw + theta
        ca, sa = math.cos(a), math.sin(a)
        d = 0.15
        while d <= LOOKAHEAD:
            for lat in (-HALF_WIDTH * 0.9, 0.0, HALF_WIDTH * 0.9):
                if self._occupied(x + d * ca - lat * sa, y + d * sa + lat * ca):
                    return d
            d += 0.1
        return LOOKAHEAD


    def process_navigation(self):
        '''
        Description:    Timer function used to drive the eBot along the route.
                        Wrapper so that nothing can raise and kill the node.

        Args:
        Returns:
        '''

        try:
            self._navigate()
        except Exception as e:
            self.get_logger().error(f"nav error: {e}", throttle_duration_sec=1.0)
            self.cmd_pub.publish(Twist())       # never leave the base coasting


    def _navigate(self):
        '''
        Description:    One control cycle: pick a safe steering direction towards the
                        current waypoint and publish a Twist.
        '''

        ############ ADD YOUR CODE HERE ############

        # route finished: keep the base stopped (it holds its last command otherwise)
        if self.done:
            self.cmd_pub.publish(Twist())
            return

        # return early until route, odometry and scan have all arrived
        if self.route is None or self.odom is None or self.scan is None:
            return

        # where we are and where we are heading
        x, y, yaw = self.odom
        last = len(self.route) - 1
        tx, ty = self.route[self.wp_idx]
        distance = math.hypot(tx - x, ty - y)

        # is this waypoint done? (in order, one at a time)
        if distance < (LAST_TOL if self.wp_idx == last else WP_TOL):
            self.get_logger().info(f"Waypoint {self.wp_idx + 1}/{len(self.route)} reached")
            if self.wp_idx == last:
                self.done = True
                self.cmd_pub.publish(Twist())
                return
            self.wp_idx += 1
            tx, ty = self.route[self.wp_idx]
            distance = math.hypot(tx - x, ty - y)

        # direction of the waypoint relative to the base
        goal = self._wrap(math.atan2(ty - y, tx - x) - yaw)

        # how much clear room do we want in a direction before we call it open
        safe = max(0.5, min(1.0, distance))

        # free distance in a direction = min(lidar, map)
        def free_of(theta):
            return min(self._lidar_free(theta), self._map_free(x, y, yaw, theta))

        # 1) goal direction open? go straight for it
        goal_free = free_of(goal)
        if goal_free >= safe:
            theta, free = goal, goal_free
        else:
            # 2) otherwise take the open direction closest to the goal (with a little hysteresis)
            best = None
            best_free = None
            best_cost = 1e9
            most_free = -1.0
            most_theta = goal
            for deg in range(-90, 91, 6):
                th = math.radians(deg)
                f = free_of(th)
                if f > most_free:
                    most_free, most_theta = f, th
                if f >= safe:
                    cost = abs(self._wrap(th - goal)) + 0.3 * abs(self._wrap(th - self.prev_theta))
                    if cost < best_cost:
                        best, best_free, best_cost = th, f, cost
            if best is not None:
                theta, free = best, best_free
            else:
                theta, free = most_theta, most_free       # boxed in: turn towards the most open way
        self.prev_theta = theta

        # steer: rotate if far off, otherwise drive slower the worse the heading error is
        w = KP_HEADING * theta
        if abs(theta) > 0.6:
            v = 0.0
        else:
            v = V_CRUISE * max(0.0, math.cos(theta))

        # slow down as obstacles get close, stop (and just turn) when nearly touching
        v = min(v, max(0.0, free - 0.35) * 0.8)

        # ease into the final stop
        if self.wp_idx == last:
            v = min(v, max(0.08, 0.6 * distance))

        # skid steer needs a minimum turn rate to turn on the spot at all
        if v < 0.03 and abs(theta) > 0.05:
            w = math.copysign(max(abs(w), MIN_TURN), theta)

        # keep the command inside the limits
        v = max(-max_linear_mps, min(max_linear_mps, v))
        w = max(-max_angular_rps, min(max_angular_rps, w))

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"wp{self.wp_idx + 1}: dist={distance:.2f} goal={math.degrees(goal):.0f} "
            f"steer={math.degrees(theta):.0f} free={free:.2f} v={v:.2f} w={w:.2f}",
            throttle_duration_sec=1.0)

        ############################################


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    ebot_nav class to perform its task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('ebot_nav_process')                    # creating ROS node

    node.get_logger().info('Node created: eBot navigation process') # logging information

    ebot_nav_class = ebot_nav()                                     # creating a new object for class 'ebot_nav'

    try:
        rclpy.spin(ebot_nav_class)                                  # spining on the object to make it alive in ROS 2 DDS
    except KeyboardInterrupt:
        pass
    finally:
        try:
            ebot_nav_class.cmd_pub.publish(Twist())                 # stop the base on exit
        except Exception:
            pass
        ebot_nav_class.destroy_node()                               # destroy node after spin ends
        if rclpy.ok():
            rclpy.shutdown()                                        # shutdown process


if __name__ == '__main__':

    main()