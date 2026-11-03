#!/usr/bin/env python3
import sys
import math

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32

# ---------------------------------------------------------------- constants
waypoints = [
    (-0.4085, -0.5379, 0.1967),   # 1
    (-0.8000, -0.0005, 0.3967),   # 2
    (-0.7430,  0.5280, 0.1967),   # 3
    (-0.4097,  0.5280, 0.1967),   # 4
    (-0.0763,  0.5280, 0.1967),   # 5
]

servo_ns = '/ur_arm_controller'
twist_controller = 'delta_twist_controller'
joint_controller = 'delta_joint_controller'
base_frame = 'base_link'

joint_names = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

# Transit pose (elbow mid-range). The arm passes through it between HOME and every waypoint.
UNFOLDED = [0.0, -1.20, -1.60, -1.90, 1.57, 1.57]

# Documented spawn pose. Used ONLY to warn if the arm is not where it should start.
# The real HOME is whatever pose the arm is in when the start delay ends.
SPAWN_EXPECTED = [0.0, -0.523599, -2.443461, -0.174533, 1.570796, 1.570796]

# ---------------------------------------------------------------- tuning
START_DELAY_S = 3.0        # no movement for this long, so the evaluator can read the start pose

# Cartesian
CRUISE_MPS = 0.05          # final approach speed (cap is 0.15)
TRANSIT_MPS = 0.06         # speed towards the approach point
SLOW_MPS = 0.03            # speed while status 10 / 11 is active
MIN_MPS = 0.010            # floor so the taper never crawls to a stop
KP_LIN = 1.5               # speed = min(cruise, KP * distance)

POS_TOL = 0.015            # m, default "reached" tolerance (limit is 0.03)
POS_TOL_WP = {             # per-waypoint override, key = waypoint number (1..5)
    3: 0.025,              # WP3 is at full stretch (0.90 m)
}
SING_GUARD_TOL = 0.028     # m: with status 10/11 active, accept the point inside this distance

APPROACH_TOL = 0.030       # m, tolerance on the intermediate approach point
APPROACH_DZ = 0.08         # m above low waypoints
APPROACH_Z_MAX = 0.30      # waypoints with z below this get an approach point
HOLD_S = 2.0

# Stall handling
STALL_WINDOW_S = 2.0
STALL_PROGRESS = 0.002     # m of progress required per window
STALL_NEAR = 0.06          # m, only a stall when this close
STALL_ACCEPT = 0.027       # m, accept a stalled point if within this
MAX_RETRIES = 2

# Joint space
JOINT_MAX = 0.25           # rad/s per joint (cap is 0.35)
JOINT_MIN = 0.03           # rad/s: the fastest joint never goes slower than this
UNFOLD_TOL = 0.05          # rad
HOME_TOL = 0.02            # rad

# Pan swing
SAFE_RADIUS = 0.35         # m: paths passing closer than this to the base axis swing first
SWING_TOL = 0.05           # rad
SWING_MIN = 0.08
SWING_MAX = 0.25

# Optional tool axis to hold in base_link, e.g. (0, 0, -1). None = angular twist 0.
TOOL_AXIS_TARGET = None
KP_ANG = 1.0
ANG_MAX = 0.30


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v, lim):
    return max(-lim, min(lim, v))


def seg_dist_to_origin(ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(ax, ay)
    t = max(0.0, min(1.0, -(ax * dx + ay * dy) / L2))
    return math.hypot(ax + t * dx, ay + t * dy)


class arm_waypoints(Node):

    def __init__(self):
        super().__init__(
            'arm_waypoints_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)

        self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.create_subscription(JointState, '/joint_states', self.jointstatecb, 50)
        self.create_subscription(Int32, '/arm_status', self.armstatuscb, 10)

        self.switch_cli = self.create_client(
            SwitchController, f'{servo_ns}/switch_controller')

        self.tcp_pose = None
        self.joint_angles = None
        self.arm_status = None
        self.last_status = None
        self.home = None                 # joint pose at the end of the start delay

        self.state = 'WAIT'
        self.wait_t0 = None
        self.idx = 0                     # next / current waypoint (0-based)
        self.cur_wp = 1                  # waypoint number used in logs
        self.hold_start = None

        self.active_controller = None
        self.switch_future = None
        self.switch_name = None

        # joint path
        self.joint_queue = []
        self.pending = None              # 'GO' (start waypoint) or 'NEXT' (waypoint finished)

        # tcp path
        self.tcp_targets = []
        self.retries = 0
        self.accepted = False
        self.stall_t = None
        self.stall_d = None

        # swing
        self.target_az = 0.0
        self.pan_sign = 1.0
        self.swing_t0 = None
        self.swing_e0 = None

        self.last_log_t = 0.0

        self.timer = self.create_timer(0.05, self.process_waypoints)
        self.get_logger().info('Task 1B waypoint node started')
        self.get_logger().info(
            f'Waiting {START_DELAY_S:.1f} s with no movement, saving the arm pose as HOME, then per '
            f'waypoint: HOME -> unfold -> [swing] -> WPn -> hold -> unfold -> HOME')

    # ------------------------------------------------------------ callbacks
    def tcpposecb(self, data):
        self.tcp_pose = data.pose

    def jointstatecb(self, data):
        try:
            self.joint_angles = [data.position[data.name.index(j)] for j in joint_names]
        except ValueError:
            pass

    def armstatuscb(self, data):
        self.arm_status = data.data
        if data.data != self.last_status:
            self.last_status = data.data
            self.get_logger().info(f'Arm status: {data.data}')
            if data.data == 11 and self.tcp_pose is not None:
                p = self.tcp_pose.position
                self.get_logger().warn(
                    f'WARNING_SINGULARITY (advisory) during {self.state} at WP{self.cur_wp}, '
                    f'tcp=({p.x:.3f},{p.y:.3f},{p.z:.3f}) radius={math.hypot(p.x, p.y):.3f} m, '
                    f'joints={self.fmt(self.joint_angles)}')
        if 20 <= data.data <= 23 and self.state != 'LATCHED':
            self.get_logger().error(
                f'LATCHED status {data.data} during {self.state} at WP{self.cur_wp}. '
                f'Joints={self.fmt(self.joint_angles)}. Stopping.')
            self.state = 'LATCHED'

    # ------------------------------------------------------------ helpers
    def fmt(self, q):
        return 'None' if q is None else '[' + ', '.join(f'{v:.3f}' for v in q) + ']'

    def now_s(self):
        return self.get_clock().now().nanoseconds / 1e9

    def tol_for(self, idx):
        """Reached-tolerance for waypoint idx (0-based)."""
        return POS_TOL_WP.get(idx + 1, POS_TOL)

    def ensure_controller(self, name):
        """Non-blocking. True once `name` is the active controller."""
        if self.active_controller == name:
            return True
        if self.switch_future is None:
            if not self.switch_cli.service_is_ready():
                return False
            req = SwitchController.Request()
            req.activate_controllers = [name]
            req.deactivate_controllers = []
            req.strictness = SwitchController.Request.STRICT
            req.activate_asap = False
            req.timeout.sec = 0
            req.timeout.nanosec = 0
            self.get_logger().info(f'Switching to {name}...')
            self.switch_name = name
            self.switch_future = self.switch_cli.call_async(req)
            return False
        if not self.switch_future.done():
            return False
        try:
            res = self.switch_future.result()
        except Exception as e:
            self.get_logger().error(f'Switch failed: {e}')
            res = None
        self.switch_future = None
        if res is not None and res.ok:
            self.active_controller = self.switch_name
            self.get_logger().info(f'{self.switch_name} active')
        else:
            self.get_logger().warn('Switch not accepted, will retry')
        return self.active_controller == name

    def pub_twist(self, vx, vy, vz, wx=0.0, wy=0.0, wz=0.0):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = base_frame
        m.twist.linear.x, m.twist.linear.y, m.twist.linear.z = vx, vy, vz
        m.twist.angular.x, m.twist.angular.y, m.twist.angular.z = wx, wy, wz
        self.twist_pub.publish(m)

    def pub_joints(self, vels):
        m = JointJog()
        m.header.stamp = self.get_clock().now().to_msg()
        m.joint_names = list(joint_names)
        m.velocities = [float(v) for v in vels]
        self.joint_pub.publish(m)

    def tool_axis(self):
        q = self.tcp_pose.orientation
        return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()[:, 2]

    def angular_cmd(self):
        if TOOL_AXIS_TARGET is None:
            return 0.0, 0.0, 0.0
        a = self.tool_axis()
        b = np.array(TOOL_AXIS_TARGET, dtype=float)
        b /= np.linalg.norm(b)
        c = np.cross(a, b)
        n = float(np.linalg.norm(c))
        if n < 1e-6:
            return 0.0, 0.0, 0.0
        e = c / n * math.atan2(n, float(np.dot(a, b)))
        w = KP_ANG * e
        mag = float(np.linalg.norm(w))
        if mag > ANG_MAX:
            w *= ANG_MAX / mag
        return float(w[0]), float(w[1]), float(w[2])

    def drive_to(self, ex, ey, ez, dist, cruise):
        if self.arm_status in (10, 11):
            cruise = min(cruise, SLOW_MPS)
        speed = max(MIN_MPS, min(cruise, KP_LIN * dist))
        s = speed / dist
        wx, wy, wz = self.angular_cmd()
        self.pub_twist(ex * s, ey * s, ez * s, wx, wy, wz)

    # ------------------------------------------------------------ joint paths
    def start_joint_path(self, queue, pending):
        self.joint_queue = queue
        self.pending = pending
        self.state = 'JOINT'

    def do_joint(self):
        """Joint-space moves. Each joint is clamped on its own, as in the version that worked."""
        if not self.ensure_controller(joint_controller):
            return
        if not self.joint_queue:
            self.joint_done()
            return
        label, target, tol = self.joint_queue[0]
        errs = [target[i] - self.joint_angles[i] for i in range(6)]
        maxe = max(abs(e) for e in errs)
        if maxe <= tol:
            self.pub_joints([0.0] * 6)
            self.get_logger().info(f'{label} pose reached. joints={self.fmt(self.joint_angles)}')
            self.joint_queue.pop(0)
            if not self.joint_queue:
                self.joint_done()
            return
        vels = [clamp(e, JOINT_MAX) for e in errs]
        peak = max(abs(v) for v in vels)
        if 1e-9 < peak < JOINT_MIN:
            k = JOINT_MIN / peak
            vels = [v * k for v in vels]
        self.pub_joints(vels)

    def joint_done(self):
        if self.pending == 'GO':
            self.begin_waypoint()
        elif self.pending == 'NEXT':
            if self.idx >= len(waypoints):
                self.get_logger().info('ALL WAYPOINTS COMPLETED, arm back at HOME')
                self.state = 'DONE'
            else:
                self.start_joint_path([('unfold', UNFOLDED, UNFOLD_TOL)], 'GO')

    # ------------------------------------------------------------ waypoint flow
    def build_tcp_targets(self):
        tx, ty, tz = waypoints[self.idx]
        targets = []
        if tz < APPROACH_Z_MAX:
            targets.append(((tx, ty, tz + APPROACH_DZ), APPROACH_TOL, False))
        targets.append(((tx, ty, tz), self.tol_for(self.idx), True))
        return targets

    def begin_waypoint(self):
        """Called with the arm at the unfold pose."""
        self.cur_wp = self.idx + 1
        self.hold_start = None
        self.retries = 0
        self.accepted = False
        self.stall_t = None
        self.tcp_targets = self.build_tcp_targets()
        tx, ty, tz = waypoints[self.idx]
        p = self.tcp_pose.position
        d = seg_dist_to_origin(p.x, p.y, tx, ty)
        self.target_az = math.atan2(ty, tx)
        self.swing_t0 = None
        if d < SAFE_RADIUS:
            self.get_logger().info(
                f'WP{self.cur_wp}: straight path passes {d:.2f} m from base axis '
                f'-> swinging shoulder_pan first (azimuth {math.degrees(self.target_az):.0f} deg)')
            self.state = 'SWING'
        else:
            self.get_logger().info(f'WP{self.cur_wp}: direct TCP move (path clear by {d:.2f} m)')
            self.state = 'TCP'

    def do_swing(self):
        if not self.ensure_controller(joint_controller):
            return
        p = self.tcp_pose.position
        err = wrap(self.target_az - math.atan2(p.y, p.x))
        if abs(err) < SWING_TOL:
            self.pub_joints([0.0] * 6)
            self.get_logger().info('Swing done -> TCP move')
            self.state = 'TCP'
            return
        t = self.now_s()
        if self.swing_t0 is None:
            self.swing_t0, self.swing_e0 = t, abs(err)
        elif t - self.swing_t0 > 1.0:
            if abs(err) > self.swing_e0 + 0.03:
                self.pan_sign *= -1.0
                self.get_logger().warn('Pan swing was going the wrong way, flipping sign')
            self.swing_t0, self.swing_e0 = t, abs(err)
        mag = max(SWING_MIN, min(SWING_MAX, abs(err)))
        v = math.copysign(mag, err) * self.pan_sign
        self.pub_joints([v, 0.0, 0.0, 0.0, 0.0, 0.0])

    def finish_waypoint(self, dist):
        self.get_logger().info(
            f'WP{self.cur_wp} hold complete (final error {dist * 1000:.1f} mm) '
            f'-> unfold -> HOME')
        self.idx += 1
        self.hold_start = None
        self.pub_twist(0.0, 0.0, 0.0)
        self.start_joint_path(
            [('unfold', UNFOLDED, UNFOLD_TOL), ('home', self.home, HOME_TOL)], 'NEXT')

    def do_tcp(self):
        if not self.ensure_controller(twist_controller):
            return
        (tx, ty, tz), tol, final = self.tcp_targets[0]
        p = self.tcp_pose.position
        ex, ey, ez = tx - p.x, ty - p.y, tz - p.z
        dist = math.sqrt(ex * ex + ey * ey + ez * ez)
        now = self.now_s()

        if now - self.last_log_t > 1.0:
            self.last_log_t = now
            self.get_logger().info(
                f'WP{self.cur_wp}{"" if final else " (approach)"}: dist={dist:.3f} m, '
                f'radius={math.hypot(p.x, p.y):.3f} m, status={self.arm_status}')

        # ---- intermediate approach point
        if not final:
            if dist <= tol:
                self.tcp_targets.pop(0)
                return
            self.drive_to(ex, ey, ez, max(dist, 1e-6), TRANSIT_MPS)
            return

        # ---- singularity guard: a warning is active and we are already inside 3 cm,
        #      so stop here rather than push further towards a latch
        if (not self.accepted and self.arm_status in (10, 11) and dist <= SING_GUARD_TOL):
            self.get_logger().warn(
                f'WP{self.cur_wp}: status {self.arm_status} with {dist * 1000:.1f} mm left, '
                f'accepting the point to avoid a latch. joints={self.fmt(self.joint_angles)}')
            self.accepted = True

        # ---- final waypoint: reached / holding
        if dist <= tol or self.accepted:
            self.pub_twist(0.0, 0.0, 0.0)
            if self.hold_start is None:
                self.hold_start = now
                self.get_logger().info(
                    f'WP{self.cur_wp} reached (error {dist * 1000:.1f} mm), holding {HOLD_S:.1f} s')
            elif now - self.hold_start >= HOLD_S:
                self.finish_waypoint(dist)
            return

        # ---- stall detection
        if self.stall_t is None:
            self.stall_t, self.stall_d = now, dist
        elif now - self.stall_t >= STALL_WINDOW_S:
            progress = self.stall_d - dist
            if dist < STALL_NEAR and progress < STALL_PROGRESS:
                self.get_logger().warn(
                    f'WP{self.cur_wp} stalled at {dist * 1000:.1f} mm, status={self.arm_status}, '
                    f'joints={self.fmt(self.joint_angles)}')
                if dist <= STALL_ACCEPT:
                    self.get_logger().info('Within 3 cm, accepting the point')
                    self.accepted = True
                    return
                if self.retries < MAX_RETRIES:
                    self.retries += 1
                    self.get_logger().warn(
                        f'Retry {self.retries}/{MAX_RETRIES}: backing up and re-approaching from above')
                    tx_, ty_, tz_ = waypoints[self.idx]
                    self.tcp_targets = [
                        ((tx_, ty_, tz_ + APPROACH_DZ + 0.04), APPROACH_TOL, False),
                        ((tx_, ty_, tz_), self.tol_for(self.idx), True)]
                else:
                    self.get_logger().error(
                        f'WP{self.cur_wp}: could not get under 3 cm, continuing anyway')
                    self.accepted = True
            self.stall_t, self.stall_d = now, dist

        self.drive_to(ex, ey, ez, dist, CRUISE_MPS)

    # ------------------------------------------------------------ main timer
    def process_waypoints(self):
        if self.tcp_pose is None or self.joint_angles is None or self.arm_status is None:
            return
        if self.state in ('LATCHED', 'DONE'):
            self.pub_twist(0.0, 0.0, 0.0)
            return
        if self.state == 'WAIT':
            # nothing is published during the wait
            if self.wait_t0 is None:
                self.wait_t0 = self.now_s()
                self.get_logger().info(
                    f'Arm data received. No movement for {START_DELAY_S:.1f} s. '
                    f'joints={self.fmt(self.joint_angles)}')
            elif self.now_s() - self.wait_t0 >= START_DELAY_S:
                self.home = list(self.joint_angles)
                self.get_logger().info(f'HOME saved: {self.fmt(self.home)}')
                dev = max(abs(a - b) for a, b in zip(self.home, SPAWN_EXPECTED))
                if dev > 0.05:
                    self.get_logger().warn(
                        f'HOME is {dev:.2f} rad away from the documented spawn pose. '
                        f'The arm was already moved: stop old nodes and relaunch the world, '
                        f'otherwise the evaluator will refuse this run.')
                self.start_joint_path([('unfold', UNFOLDED, UNFOLD_TOL)], 'GO')
            return
        if self.state == 'JOINT':
            self.do_joint()
        elif self.state == 'SWING':
            self.do_swing()
        elif self.state == 'TCP':
            self.do_tcp()


def main():
    rclpy.init(args=sys.argv)
    node = arm_waypoints()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.pub_twist(0.0, 0.0, 0.0)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():                      # Ctrl-C may already have shut rclpy down
            rclpy.shutdown()


if __name__ == '__main__':
    main()