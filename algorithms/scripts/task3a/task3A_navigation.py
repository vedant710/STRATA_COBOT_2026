#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 3A of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:		    task3A_navigation.py
# Functions:
#			        [ pathcb, odomcb, process_navigation, _navigate, _stop,
#                     _wrap, _yaw_of, _clamp, main ]
# Nodes:
#			        ebot_nav_node
#			        Publishing Topics  - [ /cmd_vel ]
#                   Subscribing Topics - [ /ebot_path, /odom ]
#
# Change from Task 2B: two speed profiles. The leg that starts at the ore drop pose carries
# the ore package, so it keeps a gentle acceleration and braking so the package does not
# slide on the deck. The two legs without the package (home -> ore drop pose, and arm ->
# home) drive at nearly the 0.5 m/s limit with a stronger acceleration and a later braking.


################### IMPORT MODULES #######################

import rclpy
import sys
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


##################### TASK CONSTANTS #######################

path_topic = '/ebot_path'
odom_topic = '/odom'
cmd_topic = '/cmd_vel'

max_linear_mps = 0.5
max_angular_rps = 1.0

# ---- controller tuning ----
CONTROL_DT = 0.05        # s: control period
V_MIN_END = 0.08
LOOKAHEAD = 0.40
KP_HEADING = 1.5
MIN_TURN = 0.30
TURN_IN_PLACE = 0.70
TURN_RESUME = 0.25
GOAL_TOL = 0.12
YAW_DONE = 0.05
SEARCH_WINDOW = 40

# ---- speed profile: empty deck (leg 1 and leg 3) ----
V_FREE = 0.48            # m/s, just under the 0.5 m/s limit
ACCEL_FREE = 0.90        # m/s^2
BRAKE_FREE = 0.90        # speed <= BRAKE * distance to the end point

# ---- speed profile: carrying the ore package (leg 2) ----
V_CARRY = 0.45           # m/s
ACCEL_CARRY = 0.40       # m/s^2: the value that kept the package on the deck in Task 3A
BRAKE_CARRY = 0.60

# The leg that starts within CARRY_RADIUS of the ore drop pose carries the package.
DROP_POSE = (6.0, 1.9)
CARRY_RADIUS = 1.0


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

        super().__init__(
            'ebot_nav_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        ############ Topic SUBSCRIPTIONS ############

        latched = QoSProfile(depth=1,
                             history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_sub = self.create_subscription(Path, path_topic, self.pathcb, latched)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)

        ############ Topic PUBLISHERS ############

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        self.timer = self.create_timer(CONTROL_DT, self.process_navigation)

        self.route = None
        self.odom = None

        ############ ADD YOUR CODE HERE ############

        self.goal_yaw = 0.0
        self.idx = 0
        self.phase = 'track'
        self.rotating = False
        self.done = True
        self.v_prev = 0.0            # last forward speed commanded (for the acceleration limit)

        # speed profile of the current leg (set in pathcb)
        self.v_cruise = V_FREE
        self.accel = ACCEL_FREE
        self.brake = BRAKE_FREE

        ############################################


    def pathcb(self, data):
        '''
        Description:    Callback function for the route topic.
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

        self.idx = 0
        self.phase = 'track'
        self.rotating = False
        self.done = False
        self.v_prev = 0.0

        # the leg that starts at the ore drop pose carries the package
        sx, sy = self.route[0]
        carrying = math.hypot(sx - DROP_POSE[0], sy - DROP_POSE[1]) < CARRY_RADIUS
        if carrying:
            self.v_cruise, self.accel, self.brake = V_CARRY, ACCEL_CARRY, BRAKE_CARRY
        else:
            self.v_cruise, self.accel, self.brake = V_FREE, ACCEL_FREE, BRAKE_FREE

        self.get_logger().info(
            f"New path: {len(self.route)} points, final yaw {self.goal_yaw:.2f} rad, "
            f"frame {data.header.frame_id}, "
            f"{'carrying the package' if carrying else 'empty deck'}: "
            f"cruise {self.v_cruise:.2f} m/s, accel {self.accel:.2f} m/s^2")

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
        self.v_prev = 0.0
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
            self._stop()


    def _navigate(self):
        '''
        Description:    One control cycle: follow the current path with a lookahead
                        point, then stop and rotate to the final yaw.
        '''

        ############ ADD YOUR CODE HERE ############

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
            self.v_prev = 0.0
            self.cmd_pub.publish(cmd)
            return

        # ---------- tracking phase ----------
        hi = min(last, self.idx + SEARCH_WINDOW)
        best_i, best_d = self.idx, float('inf')
        for i in range(self.idx, hi + 1):
            d = math.hypot(pts[i][0] - x, pts[i][1] - y)
            if d < best_d:
                best_i, best_d = i, d
        self.idx = best_i

        j = self.idx
        while j < last and math.hypot(pts[j][0] - x, pts[j][1] - y) < LOOKAHEAD:
            j += 1
        tx, ty = pts[j]

        alpha = self._wrap(math.atan2(ty - y, tx - x) - yaw)

        if abs(alpha) > TURN_IN_PLACE:
            self.rotating = True
        elif abs(alpha) < TURN_RESUME:
            self.rotating = False

        if self.rotating:
            v = 0.0
        else:
            v = self.v_cruise * max(0.0, math.cos(alpha))
            v = min(v, max(V_MIN_END, self.brake * dist_end))
            # acceleration limit: speed may rise by accel * dt per cycle (slowing is free)
            v = min(v, self.v_prev + self.accel * CONTROL_DT)

        w = self._clamp(KP_HEADING * alpha, -max_angular_rps, max_angular_rps)

        if v < 0.03 and abs(alpha) > 0.05:
            w = math.copysign(max(abs(w), MIN_TURN), alpha)

        v = self._clamp(v, -max_linear_mps, max_linear_mps)
        self.v_prev = v

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

    rclpy.init(args=sys.argv)

    node = rclpy.create_node('ebot_nav_process')

    node.get_logger().info('Node created: eBot navigation process')

    ebot_nav_class = ebot_nav()

    try:
        rclpy.spin(ebot_nav_class)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            ebot_nav_class.cmd_pub.publish(Twist())
        except Exception:
            pass
        ebot_nav_class.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':

    main()