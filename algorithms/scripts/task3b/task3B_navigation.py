#!/usr/bin/env python3

# Filename: task3B_navigation.py
# Nodes:    ebot_nav_node
#           Publishing Topics  - [ /cmd_vel, /ebot_nav/step_done ]
#           Subscribing Topics - [ /ebot_path, /odom ]
#
# Changes from Task 3A:
#   - noisy odometry is low-pass filtered
#   - stops closer to the end point (GOAL_TOL 0.07 m) so the eBot rests on its pose
#   - shorter lookahead, and slower in bends or when off the path, so corners are not cut
#   - publishes /ebot_nav/step_done (True when a step has finished, False while driving)
#     so the planner does not have to guess from noisy odometry


import rclpy
import sys
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


##################### TASK CONSTANTS #######################

path_topic = '/ebot_path'
odom_topic = '/odom'
cmd_topic = '/cmd_vel'
done_topic = '/ebot_nav/step_done'

max_linear_mps = 0.5
max_angular_rps = 1.0

# ---- controller tuning ----
CONTROL_DT = 0.05
V_MIN_END = 0.08
LOOKAHEAD = 0.32            # was 0.40: cuts corners less
KP_HEADING = 1.5
MIN_TURN = 0.30
TURN_IN_PLACE = 0.70
TURN_RESUME = 0.25
SEARCH_WINDOW = 40

# bends and drift: slow down so the base stays on the path
BEND_GAIN = 0.8             # v *= max(BEND_MIN, 1 - BEND_GAIN * |alpha|)
BEND_MIN = 0.55
OFF_PATH_SLOW = 0.25        # m: beyond this distance from the path, halve the speed

# ---- end point ----
GOAL_TOL = 0.07             # was 0.12: stop closer to the end pose
GOAL_LOST = 0.20            # in the final phase, drift beyond this goes back to tracking
KP_FINAL = 2.0
MIN_TURN_FINAL = 0.25
YAW_DONE = 0.08
DONE_CYCLES = 4             # consecutive good cycles needed to finish

# ---- odometry filter ----
FILTER_ALPHA = 0.4          # 1.0 = no filtering, smaller = smoother
JUMP_RESET = 0.5            # m: bigger jump = teleport (new world), restart the filter

# ---- speed profile: empty deck ----
V_FREE = 0.48
ACCEL_FREE = 0.90
BRAKE_FREE = 0.90

# ---- speed profile: carrying the ore package ----
V_CARRY = 0.45
ACCEL_CARRY = 0.40
BRAKE_CARRY = 0.60

DROP_POSE = (6.0, 1.9)
CARRY_RADIUS = 1.0


##################### CLASS DEFINITION #######################

class ebot_nav(Node):
    '''
    Drives the eBot along each path published on /ebot_path, stops at its end pose,
    reports the step finished, and waits for the next path.
    '''

    def __init__(self):
        super().__init__(
            'ebot_nav_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        latched = QoSProfile(depth=1,
                             history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_sub = self.create_subscription(Path, path_topic, self.pathcb, latched)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.done_pub = self.create_publisher(Bool, done_topic, latched)

        self.timer = self.create_timer(CONTROL_DT, self.process_navigation)

        self.route = None
        self.odom = None

        self.goal_yaw = 0.0
        self.idx = 0
        self.phase = 'track'
        self.rotating = False
        self.done = True
        self.v_prev = 0.0
        self.ok_count = 0

        # odometry filter state
        self.fx = None
        self.fy = None
        self.fc = 1.0
        self.fs = 0.0

        self.v_cruise = V_FREE
        self.accel = ACCEL_FREE
        self.brake = BRAKE_FREE

        self.publish_done(False)


    def publish_done(self, value):
        '''Tell the planner whether the current step has finished.'''
        msg = Bool()
        msg.data = bool(value)
        self.done_pub.publish(msg)


    def pathcb(self, data):
        '''Every new path replaces the old one and restarts the follower.'''
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
        self.ok_count = 0
        self.publish_done(False)

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


    def odomcb(self, data):
        '''Stores a low-pass filtered pose (yaw filtered through cos / sin).'''
        p = data.pose.pose.position
        yaw = self._yaw_of(data.pose.pose.orientation)

        if self.fx is None or math.hypot(p.x - self.fx, p.y - self.fy) > JUMP_RESET:
            self.fx, self.fy = p.x, p.y
            self.fc, self.fs = math.cos(yaw), math.sin(yaw)
        else:
            a = FILTER_ALPHA
            self.fx += a * (p.x - self.fx)
            self.fy += a * (p.y - self.fy)
            self.fc += a * (math.cos(yaw) - self.fc)
            self.fs += a * (math.sin(yaw) - self.fs)

        self.odom = (self.fx, self.fy, math.atan2(self.fs, self.fc))


    # ------------------------- helpers -------------------------

    @staticmethod
    def _wrap(e):
        return math.atan2(math.sin(e), math.cos(e))

    @staticmethod
    def _yaw_of(q):
        return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    @staticmethod
    def _clamp(v, low, high):
        return max(low, min(high, v))

    def _stop(self):
        self.v_prev = 0.0
        self.cmd_pub.publish(Twist())


    def process_navigation(self):
        '''Timer wrapper so that nothing can raise and kill the node.'''
        try:
            self._navigate()
        except Exception as e:
            self.get_logger().error(f"nav error: {e}", throttle_duration_sec=1.0)
            self._stop()


    def _navigate(self):
        '''One control cycle: follow the path with a lookahead point, then align the yaw.'''
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
            self.ok_count = 0
            self.get_logger().info(f"Reached the end point (dist {dist_end:.2f} m), aligning yaw")

        if self.phase == 'final' and dist_end > GOAL_LOST:
            self.phase = 'track'                    # drifted away while turning: drive back
            self.get_logger().info(f"Drifted {dist_end:.2f} m from the end point, driving back")

        if self.phase == 'final':
            err = self._wrap(self.goal_yaw - yaw)

            if abs(err) < YAW_DONE:
                self.ok_count += 1
                self._stop()                        # hold still while confirming
                if self.ok_count >= DONE_CYCLES:
                    self.done = True
                    self.ok_count = 0
                    self.publish_done(True)
                    self.get_logger().info(
                        f"Step finished: x={x:.2f} y={y:.2f} yaw={yaw:.2f}, "
                        f"waiting for the next path")
                return

            self.ok_count = 0
            w = self._clamp(KP_FINAL * err, -max_angular_rps, max_angular_rps)
            w = math.copysign(max(abs(w), MIN_TURN_FINAL), err)
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
            v *= max(BEND_MIN, 1.0 - BEND_GAIN * abs(alpha))        # slower in bends
            if best_d > OFF_PATH_SLOW:
                v *= 0.5                                            # slower when off the path
            v = min(v, max(V_MIN_END, self.brake * dist_end))
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


##################### FUNCTION DEFINITION #######################

def main():
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