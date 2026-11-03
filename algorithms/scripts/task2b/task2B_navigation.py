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
# Filename:		    task2B_navigation.py
# Functions:
#			        [ pathcb, odomcb, process_navigation, _navigate, _stop,
#                     _wrap, _yaw_of, _clamp, main ]
# Nodes:
#			        ebot_nav_node
#			        Publishing Topics  - [ /cmd_vel ]
#                   Subscribing Topics - [ /ebot_path, /odom ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


##################### TASK CONSTANTS #######################

# The planner publishes each path with held-message (latched) QoS.
path_topic = '/ebot_path'
odom_topic = '/odom'
cmd_topic = '/cmd_vel'

# What /cmd_vel accepts. The base clips anything beyond these.
max_linear_mps = 0.5
max_angular_rps = 1.0

# ---- controller tuning ----
V_CRUISE = 0.30          # normal driving speed (m/s)
V_MIN_END = 0.08         # slowest creep speed while approaching the end (m/s)
LOOKAHEAD = 0.40         # m: how far ahead on the path we aim
KP_HEADING = 1.5         # heading P gain
MIN_TURN = 0.30          # minimum turn rate (rad/s); skid steer needs a push
TURN_IN_PLACE = 0.70     # rad: above this heading error, stop and rotate on the spot
TURN_RESUME = 0.25       # rad: resume driving once the error falls below this
GOAL_TOL = 0.12          # m: close enough to the last point to start the final rotation
YAW_DONE = 0.05          # rad: final heading error accepted (scoring allows 0.15)
SEARCH_WINDOW = 40       # path points searched ahead when finding our progress


##################### CLASS DEFINITION #######################

class ebot_nav(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to drive the eBot along each path
                    published on /ebot_path, stop at its end pose, and wait for the
                    next path.
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

        # latched QoS so a path published before this node started is still delivered
        latched = QoSProfile(depth=1,
                             history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_sub = self.create_subscription(Path, path_topic, self.pathcb, latched)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)

        ############ Topic PUBLISHERS ############

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)                      # the eBot drives on what you publish here

        ############ Constructor VARIABLES/OBJECTS ############

        control_rate = 0.05                                                             # rate of time to run one control cycle (seconds)
        self.timer = self.create_timer(control_rate, self.process_navigation)           # creating a timer based function which gets called on every 0.05 seconds (as defined by 'control_rate' variable)

        self.route = None                                                               # current path as [(x, y), ...] (from pathcb())
        self.odom = None                                                                # where the base is (from odomcb())

        ############ ADD YOUR CODE HERE ############

        self.goal_yaw = 0.0          # heading the eBot must end the step with
        self.idx = 0                 # index of the path point nearest to the eBot
        self.phase = 'track'         # 'track' = follow the path, 'final' = rotate to goal_yaw
        self.rotating = False        # True while turning on the spot before driving
        self.done = True             # True when there is nothing to drive (no path yet or step finished)

        ############################################


    def pathcb(self, data):
        '''
        Description:    Callback function for the route topic.
                        Use this function to receive the waypoints the eBot has to drive.
                        Every new path replaces the old one and restarts the follower.

        Args:
            data (Path):    The route, as a sequence of poses

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        if len(data.poses) < 2:
            self.get_logger().warn("Ignoring a path with fewer than 2 poses")
            return

        self.route = [(p.pose.position.x, p.pose.position.y) for p in data.poses]
        self.goal_yaw = self._yaw_of(data.poses[-1].pose.orientation)

        # restart for the new step
        self.idx = 0
        self.phase = 'track'
        self.rotating = False
        self.done = False

        self.get_logger().info(
            f"New path: {len(self.route)} points, final yaw {self.goal_yaw:.2f} rad, "
            f"frame {data.header.frame_id}")

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
        yaw = self._yaw_of(data.pose.pose.orientation)
        self.odom = (p.x, p.y, yaw)

        ############################################


    # ------------------------- helpers -------------------------

    @staticmethod
    def _wrap(e):
        '''wrap an angle to [-pi, pi]'''
        return math.atan2(math.sin(e), math.cos(e))

    @staticmethod
    def _yaw_of(q):
        '''yaw (rad) from a quaternion'''
        return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    @staticmethod
    def _clamp(v, low, high):
        '''limit v to the range [low, high]'''
        return max(low, min(high, v))

    def _stop(self):
        '''publish a zero velocity command (the base holds its last command otherwise)'''
        self.cmd_pub.publish(Twist())


    def process_navigation(self):
        '''
        Description:    Timer function used to drive the eBot along the path.
                        Wrapper so that nothing can raise and kill the node.

        Args:
        Returns:
        '''

        try:
            self._navigate()
        except Exception as e:
            self.get_logger().error(f"nav error: {e}", throttle_duration_sec=1.0)
            self._stop()                        # never leave the base coasting


    def _navigate(self):
        '''
        Description:    One control cycle: follow the current path with a lookahead
                        point, then stop and rotate to the final yaw.
        '''

        ############ ADD YOUR CODE HERE ############

        # nothing to drive yet, or the step is finished: stay stopped
        if self.route is None or self.odom is None or self.done:
            if self.odom is not None:
                self._stop()
            return

        x, y, yaw = self.odom
        pts = self.route
        last = len(pts) - 1
        gx, gy = pts[last]
        dist_end = math.hypot(gx - x, gy - y)

        # ---------- final phase: stand at the end point and rotate to the goal yaw ----------
        if self.phase == 'track' and dist_end < GOAL_TOL:
            self.phase = 'final'
            self.get_logger().info(f"Reached the end point (dist {dist_end:.2f} m), aligning yaw")

        if self.phase == 'final':
            err = self._wrap(self.goal_yaw - yaw)
            if abs(err) < YAW_DONE:
                self.done = True
                self._stop()
                self.get_logger().info(
                    f"Step finished: x={x:.2f} y={y:.2f} yaw={yaw:.2f}, waiting for the next path")
                return
            w = self._clamp(KP_HEADING * err, -max_angular_rps, max_angular_rps)
            w = math.copysign(max(abs(w), MIN_TURN), err)
            cmd = Twist()
            cmd.angular.z = float(w)
            self.cmd_pub.publish(cmd)
            return

        # ---------- tracking phase ----------
        # progress: nearest path point, searching forward only, so we never go backwards
        hi = min(last, self.idx + SEARCH_WINDOW)
        best_i, best_d = self.idx, float('inf')
        for i in range(self.idx, hi + 1):
            d = math.hypot(pts[i][0] - x, pts[i][1] - y)
            if d < best_d:
                best_i, best_d = i, d
        self.idx = best_i

        # lookahead point: first path point at least LOOKAHEAD away, from our progress on
        j = self.idx
        while j < last and math.hypot(pts[j][0] - x, pts[j][1] - y) < LOOKAHEAD:
            j += 1
        tx, ty = pts[j]

        # heading error towards the lookahead point
        alpha = self._wrap(math.atan2(ty - y, tx - x) - yaw)

        # turn on the spot when badly misaligned (hysteresis stops chattering)
        if abs(alpha) > TURN_IN_PLACE:
            self.rotating = True
        elif abs(alpha) < TURN_RESUME:
            self.rotating = False

        if self.rotating:
            v = 0.0
        else:
            v = V_CRUISE * max(0.0, math.cos(alpha))
            # ease into the end of the path
            v = min(v, max(V_MIN_END, 0.6 * dist_end))

        w = self._clamp(KP_HEADING * alpha, -max_angular_rps, max_angular_rps)

        # skid steer needs a minimum turn rate to rotate on the spot at all
        if v < 0.03 and abs(alpha) > 0.05:
            w = math.copysign(max(abs(w), MIN_TURN), alpha)

        v = self._clamp(v, -max_linear_mps, max_linear_mps)

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"idx={self.idx}/{last} off_path={best_d:.2f} dist_end={dist_end:.2f} "
            f"alpha={math.degrees(alpha):.0f} v={v:.2f} w={w:.2f}",
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