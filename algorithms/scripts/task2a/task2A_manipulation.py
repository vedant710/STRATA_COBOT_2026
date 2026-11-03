#!/usr/bin/env python3
'''
StrataCobot (SC) Theme (eYRC 2026-27) - Task 2A : MANIPULATION

Team ID:          [ Team-ID ]
Author List:      [ Names ]
Filename:         task2A_manipulation.py
Nodes:
    Publishing  - /delta_twist_cmds, /delta_joint_cmds, /ore_status, /magnet (service client)
    Subscribing - /tcp_pose_raw, /joint_states, /arm_status,
                  /ur7e/end_effector/force_magnitude, tf (ore transforms)

Flow
----
WAIT (no movement, HOME saved) -> SCAN (read ore transforms) -> for every ore:
    unfold -> hover above ore -> tool-down -> magnet ON -> slow descent until contact
    -> lift -> verify hold by force (~1.47 N) -> carry to the container of its kind
    -> release -> verify force ~0 -> lift away
-> unfold -> HOME.

Container drop points come from the Task 1B waypoints 3, 4, 5 (see CONTAINER_DROP).
'''

import sys
import math

import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32, Float32, String
from std_srvs.srv import SetBool

# ---------------------------------------------------------------- constants
ore_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']

servo_ns = '/ur_arm_controller'
twist_controller = 'delta_twist_controller'
joint_controller = 'delta_joint_controller'
base_frame = 'base_link'

joint_names = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

UNFOLDED = [0.0, -1.20, -1.60, -1.90, 1.57, 1.57]
SPAWN_EXPECTED = [0.0, -0.523599, -2.443461, -0.174533, 1.570796, 1.570796]

# ------------------------------------------------------------------------
# CONTAINERS: TCP release points in base_link, taken from the Task 1B
# waypoints that visit the containers:
#   WP3 (-0.7430, 0.5280, 0.1967) -> vanadinite  (nearest, at full stretch 0.91 m)
#   WP4 (-0.4097, 0.5280, 0.1967) -> malachite   (middle)
#   WP5 (-0.0763, 0.5280, 0.1967) -> azurite     (farthest)
# ------------------------------------------------------------------------
CONTAINER_DROP = {
    'azurite_ore':    (-0.0763, 0.5280, 0.1967),
    'malachite_ore':  (-0.4097, 0.5280, 0.1967),
    'vanadinite_ore': (-0.7430, 0.5280, 0.1967),
}
# reached-tolerance per container (WP3 is at full stretch, as in Task 1B)
DROP_TOL = {'azurite_ore': 0.020, 'malachite_ore': 0.020, 'vanadinite_ore': 0.025}
DROP_APPROACH_DZ = 0.08    # approach height above the drop point (same as Task 1B)
SING_GUARD_TOL = 0.028     # with status 10/11 active, accept the drop point inside this distance

# ---------------------------------------------------------------- tuning
START_DELAY_S = 3.0
EXPECTED_ORES = 4
SCAN_SETTLE_S = 1.5        # transforms must stay put this long before they are trusted
SCAN_TIMEOUT_S = 25.0

# Cartesian speeds (m/s, cap 0.15)
TRANSIT_MPS = 0.06
PRECONTACT_MPS = 0.04
DESCEND_MPS = 0.015
LIFT_MPS = 0.04
SLOW_MPS = 0.03            # while status 10 / 11 is active
MIN_MPS = 0.010
KP_LIN = 1.5
TARGET_TIMEOUT_S = 35.0

# Pick geometry (metres, relative to the ore's top face centre)
HOVER_DZ = 0.12
PRECONTACT_DZ = 0.03
LIFT_DZ = 0.20
HOVER_TOL = 0.015
XY_ALIGN_TOL = 0.005       # magnet disc 81 mm in a 101.6 mm face: ~10 mm margin per side
DESCEND_TOL = 0.002

# Attempt k uses entry k: how far past the face to aim, and force that counts as contact
CONTACT_DZ = [0.001, -0.002, -0.004]
CONTACT_N = [0.4, 1.0, 1.6]
PRESS_ABORT_N = 3.0        # never keep pushing beyond this
MAX_ATTEMPTS = 3

# Force readings (tool weight already subtracted)
HOLD_MIN_N = 0.9           # one ore ~1.47 N
HOLD_MAX_N = 2.2
FREE_MAX_N = 0.4           # released / nothing on tool
LOST_N = 0.5               # carrying and below this = ore dropped on the way
FORCE_ALPHA = 0.3

# Tool orientation (tool z axis pointing straight down in base_link)
TOOL_AXIS_TARGET = (0.0, 0.0, -1.0)    # None = no orientation control (as in Task 1B)
TILT_OK = 0.035            # rad (2 deg)
TILT_PROCEED = 0.10        # rad, accept after orient timeout
ORIENT_TIMEOUT_S = 25.0
KP_ANG = 1.0
ANG_MAX = 0.30

# Timing
MAGNET_SETTLE_S = 0.4
CONTACT_SETTLE_S = 0.6
HOLD_CHECK_S = 0.6
RELEASE_WAIT_S = 1.0

# Joint space
JOINT_MAX = 0.25
JOINT_MIN = 0.03
UNFOLD_TOL = 0.05
HOME_TOL = 0.02

# Pan swing
SAFE_RADIUS = 0.35
SWING_TOL = 0.05
SWING_MIN = 0.08
SWING_MAX = 0.25


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


class pick_and_place(Node):

    def __init__(self):
        super().__init__(
            'task2a_manipulation_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)
        self.status_pub = self.create_publisher(String, '/ore_status', 10)

        self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.create_subscription(JointState, '/joint_states', self.jointstatecb, 50)
        self.create_subscription(Int32, '/arm_status', self.armstatuscb, 10)
        self.create_subscription(
            Float32, '/ur7e/end_effector/force_magnitude', self.forcecb, 50)

        self.switch_cli = self.create_client(
            SwitchController, f'{servo_ns}/switch_controller')
        self.magnet_cli = self.create_client(SetBool, '/magnet')

        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # sensor data
        self.tcp_pose = None
        self.joint_angles = None
        self.arm_status = None
        self.last_status = None
        self.force = None
        self.force_f = 0.0
        self.home = None

        # state machine
        self.state = 'WAIT'
        self.wait_t0 = None
        self.active_controller = None
        self.switch_future = None
        self.switch_name = None
        self.last_log_t = 0.0

        # scan / queue
        self.scan_t0 = None
        self.scan_stable_t = None
        self.scan_prev = {}
        self.queue = []

        # current ore
        self.cur = None
        self.cur_pos = None
        self.attempt = 0
        self.picked = []
        self.results = {}

        # joint path
        self.joint_queue = []
        self.pending = None

        # tcp path
        self.tcp_targets = []
        self.after_tcp = None
        self.tgt_t0 = None
        self.hover_pos = None
        self.orient_t0 = None
        self.orient_ok_t = None

        # swing
        self.target_az = 0.0
        self.pan_sign = 1.0
        self.swing_t0 = None
        self.swing_e0 = None

        # magnet / pause
        self.magnet_future = None
        self.magnet_target = False
        self.magnet_then = None
        self.pause_until = 0.0
        self.pause_then = None

        self.timer = self.create_timer(0.05, self.process)
        self.status_timer = self.create_timer(0.5, self.publish_status)
        self.get_logger().info('Task 2A manipulation node started')

    # ------------------------------------------------------------ callbacks
    def tcpposecb(self, data):
        self.tcp_pose = data.pose

    def jointstatecb(self, data):
        try:
            self.joint_angles = [data.position[data.name.index(j)] for j in joint_names]
        except ValueError:
            pass

    def forcecb(self, data):
        self.force = float(data.data)
        self.force_f = (self.force if self.force_f is None else
                        (1.0 - FORCE_ALPHA) * self.force_f + FORCE_ALPHA * self.force)

    def armstatuscb(self, data):
        self.arm_status = data.data
        if data.data != self.last_status:
            self.last_status = data.data
            self.get_logger().info(f'Arm status: {data.data}')
        if 20 <= data.data <= 23 and self.state != 'LATCHED':
            self.get_logger().error(
                f'PROTECTIVE STOP {data.data} during {self.state} (ore {self.cur}). '
                f'Joints={self.fmt(self.joint_angles)}. Stopping. '
                f'(Unlock-and-continue is not implemented here: see "Recovering from a latch".)')
            self.state = 'LATCHED'

    def publish_status(self):
        locked = self.state not in ('WAIT', 'SCAN')
        msg = String()
        msg.data = (f'locked={1 if locked else 0};'
                    f'active={self.cur if self.cur else "-"};'
                    f'picked={",".join(self.picked)}')
        self.status_pub.publish(msg)

    # ------------------------------------------------------------ helpers
    def fmt(self, q):
        return 'None' if q is None else '[' + ', '.join(f'{v:.3f}' for v in q) + ']'

    def now_s(self):
        return self.get_clock().now().nanoseconds / 1e9

    def ore_pos(self, name):
        try:
            t = self.tf_buffer.lookup_transform(base_frame, name, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        v = t.transform.translation
        return np.array([v.x, v.y, v.z])

    def tcp_xyz(self):
        p = self.tcp_pose.position
        return np.array([p.x, p.y, p.z])

    def ensure_controller(self, name):
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

    def tilt(self):
        """Angle (rad) between the tool z axis and straight down."""
        a = self.tool_axis()
        return math.acos(max(-1.0, min(1.0, float(np.dot(a, [0.0, 0.0, -1.0])))))

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
        s = speed / max(dist, 1e-6)
        wx, wy, wz = self.angular_cmd()
        self.pub_twist(ex * s, ey * s, ez * s, wx, wy, wz)

    def pause(self, seconds, then):
        self.pause_until = self.now_s() + seconds
        self.pause_then = then
        self.state = 'PAUSE'

    def magnet(self, on, then):
        self.magnet_target = on
        self.magnet_then = then
        self.magnet_future = None
        self.state = 'MAGNET'

    # ------------------------------------------------------------ joint paths
    def start_joint_path(self, queue, pending):
        self.joint_queue = queue
        self.pending = pending
        self.state = 'JOINT'

    def do_joint(self):
        if not self.ensure_controller(joint_controller):
            return
        if not self.joint_queue:
            self.joint_done()
            return
        label, target, tol = self.joint_queue[0]
        errs = [target[i] - self.joint_angles[i] for i in range(6)]
        if max(abs(e) for e in errs) <= tol:
            self.pub_joints([0.0] * 6)
            self.get_logger().info(f'{label} pose reached. joints={self.fmt(self.joint_angles)}')
            self.joint_queue.pop(0)
            if not self.joint_queue:
                self.joint_done()
            return
        vels = [clamp(e, JOINT_MAX) for e in errs]
        peak = max(abs(v) for v in vels)
        if 1e-9 < peak < JOINT_MIN:
            vels = [v * JOINT_MIN / peak for v in vels]
        self.pub_joints(vels)

    def joint_done(self):
        if self.pending == 'SELECT':
            self.state = 'SELECT'
        elif self.pending == 'PICK':
            self.begin_pick()
        elif self.pending == 'CARRY':
            self.begin_drop_move()
        elif self.pending == 'FINISH':
            self.finish_run()

    # ------------------------------------------------------------ TCP paths
    def start_tcp_path(self, targets, after, swing_to=None):
        """targets: list of dicts {pos, tol, speed, kind}. after: called when all are done."""
        self.tcp_targets = targets
        self.after_tcp = after
        self.tgt_t0 = self.now_s()
        self.swing_t0 = None
        if swing_to is not None:
            p = self.tcp_xyz()
            d = seg_dist_to_origin(p[0], p[1], swing_to[0], swing_to[1])
            if d < SAFE_RADIUS:
                self.target_az = math.atan2(swing_to[1], swing_to[0])
                self.get_logger().info(
                    f'Path passes {d:.2f} m from base axis -> swinging pan to '
                    f'{math.degrees(self.target_az):.0f} deg first')
                self.state = 'SWING'
                return
        self.state = 'TCP'

    def tgt(self, x, y, z, tol, speed, kind='move'):
        return {'pos': (float(x), float(y), float(z)), 'tol': tol, 'speed': speed, 'kind': kind}

    def tcp_finished(self):
        fn, self.after_tcp = self.after_tcp, None
        self.pub_twist(0.0, 0.0, 0.0)
        fn()

    def do_swing(self):
        if not self.ensure_controller(joint_controller):
            return
        p = self.tcp_pose.position
        err = wrap(self.target_az - math.atan2(p.y, p.x))
        if abs(err) < SWING_TOL:
            self.pub_joints([0.0] * 6)
            self.get_logger().info('Swing done')
            self.tgt_t0 = self.now_s()
            self.state = 'TCP'
            return
        t = self.now_s()
        if self.swing_t0 is None:
            self.swing_t0, self.swing_e0 = t, abs(err)
        elif t - self.swing_t0 > 1.0:
            if abs(err) > self.swing_e0 + 0.03:
                self.pan_sign *= -1.0
                self.get_logger().warn('Pan swing going the wrong way, flipping sign')
            self.swing_t0, self.swing_e0 = t, abs(err)
        mag = max(SWING_MIN, min(SWING_MAX, abs(err)))
        self.pub_joints([math.copysign(mag, err) * self.pan_sign, 0, 0, 0, 0, 0])

    def do_tcp(self):
        if not self.ensure_controller(twist_controller):
            return
        t = self.tcp_targets[0]
        tx, ty, tz = t['pos']
        p = self.tcp_xyz()
        ex, ey, ez = tx - p[0], ty - p[1], tz - p[2]
        dist = math.sqrt(ex * ex + ey * ey + ez * ez)
        now = self.now_s()

        if now - self.last_log_t > 1.0:
            self.last_log_t = now
            self.get_logger().info(
                f'[{self.cur}] {t["kind"]}: dist={dist * 1000:.0f} mm, z={p[2]:.3f}, '
                f'force={self.force_f:.2f} N, tilt={math.degrees(self.tilt()):.1f} deg, '
                f'status={self.arm_status}')

        # contact handling while descending onto the ore
        if t['kind'] == 'descend':
            k = min(self.attempt, len(CONTACT_N) - 1)
            if self.force_f >= PRESS_ABORT_N:
                self.get_logger().warn(f'Pressing too hard ({self.force_f:.2f} N), stopping descent')
                self.tcp_targets = []
                self.tcp_finished()
                return
            if self.force_f >= CONTACT_N[k]:
                self.get_logger().info(f'Contact at {self.force_f:.2f} N, z={p[2]:.4f}')
                self.tcp_targets = []
                self.tcp_finished()
                return

        tol = t['tol']
        if t['kind'] == 'drop' and self.arm_status in (10, 11) and dist <= SING_GUARD_TOL:
            tol = max(tol, SING_GUARD_TOL)     # stop here rather than push towards a latch
        if dist <= tol:
            self.tcp_targets.pop(0)
            self.tgt_t0 = now
            if not self.tcp_targets:
                self.tcp_finished()
            return

        if now - self.tgt_t0 > TARGET_TIMEOUT_S:
            if dist < 0.03:
                self.get_logger().warn(f'Timeout at {dist * 1000:.0f} mm, accepting')
                self.tcp_targets.pop(0)
                self.tgt_t0 = now
                if not self.tcp_targets:
                    self.tcp_finished()
                return
            self.abort_ore('timeout driving to target')
            return

        self.drive_to(ex, ey, ez, dist, t['speed'])

    # ------------------------------------------------------------ run flow
    def do_wait(self):
        if self.wait_t0 is None:
            self.wait_t0 = self.now_s()
            self.get_logger().info(
                f'Arm data received. No movement for {START_DELAY_S:.1f} s. '
                f'joints={self.fmt(self.joint_angles)}')
            return
        if self.now_s() - self.wait_t0 < START_DELAY_S:
            return
        missing = [k for k, v in CONTAINER_DROP.items() if v is None]
        if missing:
            self.get_logger().error(
                f'CONTAINER_DROP not filled in for {missing}. Not moving. '
                f'Fill in the (x, y, z) release point of each container in base_link.')
            self.state = 'DONE'
            return
        self.home = list(self.joint_angles)
        self.get_logger().info(f'HOME saved: {self.fmt(self.home)}')
        dev = max(abs(a - b) for a, b in zip(self.home, SPAWN_EXPECTED))
        if dev > 0.05:
            self.get_logger().warn(
                f'HOME is {dev:.2f} rad away from the documented spawn pose. '
                f'Relaunch the world before an evaluated run.')
        self.scan_t0 = self.now_s()
        self.state = 'SCAN'

    def do_scan(self):
        now = self.now_s()
        found = {}
        for ot in ore_types:
            for i in (1, 2):
                n = f'{ot}_{i}'
                p = self.ore_pos(n)
                if p is not None:
                    found[n] = p
        changed = (set(found) != set(self.scan_prev) or
                   any(np.linalg.norm(found[n] - self.scan_prev[n]) > 0.005 for n in found))
        if changed or self.scan_stable_t is None:
            self.scan_stable_t = now
        self.scan_prev = found
        stable = now - self.scan_stable_t >= SCAN_SETTLE_S
        timed_out = now - self.scan_t0 > SCAN_TIMEOUT_S
        if stable and found and (len(found) >= EXPECTED_ORES or timed_out):
            ref = self.tcp_xyz()
            order = sorted(found, key=lambda n: np.linalg.norm(found[n][:2] - ref[:2]))
            self.queue = order
            self.get_logger().info(f'Scan done, {len(order)} ores: {order}')
            for n in order:
                p = found[n]
                self.get_logger().info(f'  {n}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})')
            self.state = 'SELECT'
        elif timed_out and not found:
            self.get_logger().error('No ore transforms received. Is task2A_perception.py running?')
            self.scan_t0 = now

    def do_select(self):
        if not self.queue:
            self.cur = None
            self.start_joint_path([('unfold', UNFOLDED, UNFOLD_TOL),
                                   ('home', self.home, HOME_TOL)], 'FINISH')
            return
        self.cur = self.queue.pop(0)
        self.get_logger().info(f'===== Processing {self.cur} =====')
        self.start_joint_path([('unfold', UNFOLDED, UNFOLD_TOL)], 'PICK')

    def finish_run(self):
        self.get_logger().info(f'RUN COMPLETE. Results: {self.results}')
        self.state = 'DONE'

    # ---- pick
    def begin_pick(self):
        p = self.ore_pos(self.cur)
        if p is None:
            self.results[self.cur] = 'no transform'
            self.get_logger().error(f'No transform for {self.cur}, skipping')
            self.state = 'SELECT'
            return
        self.cur_pos = p
        self.attempt = 0
        self.start_attempt()

    def start_attempt(self):
        p = self.ore_pos(self.cur)
        if p is not None:
            self.cur_pos = p
        x, y, z = self.cur_pos
        self.hover_pos = (x, y, z + HOVER_DZ)
        self.get_logger().info(
            f'{self.cur}: attempt {self.attempt + 1}/{MAX_ATTEMPTS}, ore at '
            f'({x:.3f}, {y:.3f}, {z:.3f})')
        self.start_tcp_path(
            [self.tgt(x, y, z + HOVER_DZ, HOVER_TOL, TRANSIT_MPS)],
            self.start_orient, swing_to=(x, y))

    def start_orient(self):
        self.orient_t0 = self.now_s()
        self.orient_ok_t = None
        self.state = 'ORIENT'

    def do_orient(self):
        if not self.ensure_controller(twist_controller):
            return
        now = self.now_s()
        tilt = self.tilt()
        hx, hy, hz = self.hover_pos
        p = self.tcp_xyz()
        ex, ey, ez = hx - p[0], hy - p[1], hz - p[2]
        dist = math.sqrt(ex * ex + ey * ey + ez * ez)

        if TOOL_AXIS_TARGET is None or tilt < TILT_OK:
            if self.orient_ok_t is None:
                self.orient_ok_t = now
            elif now - self.orient_ok_t >= 0.5 or TOOL_AXIS_TARGET is None:
                self.get_logger().info(f'Tool down (tilt {math.degrees(tilt):.1f} deg)')
                self.pub_twist(0.0, 0.0, 0.0)
                self.magnet(True, self.start_descend)
                return
        else:
            self.orient_ok_t = None

        if now - self.orient_t0 > ORIENT_TIMEOUT_S:
            if tilt < TILT_PROCEED:
                self.get_logger().warn(f'Orient timeout, tilt {math.degrees(tilt):.1f} deg, going on')
                self.pub_twist(0.0, 0.0, 0.0)
                self.magnet(True, self.start_descend)
            else:
                self.abort_ore(f'could not get tool down (tilt {math.degrees(tilt):.1f} deg)')
            return

        wx, wy, wz = self.angular_cmd()
        if dist > 0.004:
            s = max(MIN_MPS, min(0.03, KP_LIN * dist)) / dist
            self.pub_twist(ex * s, ey * s, ez * s, wx, wy, wz)
        else:
            self.pub_twist(0.0, 0.0, 0.0, wx, wy, wz)

    def start_descend(self):
        p = self.ore_pos(self.cur)
        if p is not None:
            self.cur_pos = p
        x, y, z = self.cur_pos
        k = min(self.attempt, len(CONTACT_DZ) - 1)
        self.pause(MAGNET_SETTLE_S, lambda: self.start_tcp_path([
            self.tgt(x, y, z + PRECONTACT_DZ, XY_ALIGN_TOL, PRECONTACT_MPS),
            self.tgt(x, y, z + CONTACT_DZ[k], DESCEND_TOL, DESCEND_MPS, 'descend'),
        ], self.after_descend))

    def after_descend(self):
        self.pause(CONTACT_SETTLE_S, self.start_lift)

    def start_lift(self):
        x, y, z = self.cur_pos
        self.start_tcp_path(
            [self.tgt(x, y, z + LIFT_DZ, 0.01, LIFT_MPS, 'lift')], self.after_lift)

    def after_lift(self):
        self.pause(HOLD_CHECK_S, self.check_hold)

    def check_hold(self):
        f = self.force_f
        if HOLD_MIN_N <= f <= HOLD_MAX_N:
            self.get_logger().info(f'{self.cur}: PICKED (force {f:.2f} N)')
            self.picked.append(self.cur)
            self.results[self.cur] = 'picked'
            self.start_carry()
            return
        self.get_logger().warn(f'{self.cur}: no hold (force {f:.2f} N)')
        self.attempt += 1
        if self.attempt >= MAX_ATTEMPTS:
            self.abort_ore('pick failed after all attempts')
            return
        self.magnet(False, lambda: self.pause(RELEASE_WAIT_S, self.start_attempt))

    # ---- carry and drop
    def start_carry(self):
        """Same safe sequence as Task 1B (in reverse): unfold first, swing from there."""
        ot = self.cur.rsplit('_', 1)[0]
        self.drop = CONTAINER_DROP[ot]
        self.drop_tol = DROP_TOL[ot]
        self.get_logger().info(f'{self.cur}: carrying, going to the unfold pose before the swing')
        self.start_joint_path([('carry-unfold', UNFOLDED, UNFOLD_TOL)], 'CARRY')

    def begin_drop_move(self):
        cx, cy, cz = self.drop
        self.start_tcp_path(
            [self.tgt(cx, cy, cz + DROP_APPROACH_DZ, 0.03, TRANSIT_MPS),
             self.tgt(cx, cy, cz, self.drop_tol, PRECONTACT_MPS, 'drop')],
            self.at_drop, swing_to=(cx, cy))

    def at_drop(self):
        self.pause(HOLD_CHECK_S, self.release_now)

    def release_now(self):
        if self.force_f < LOST_N:
            self.get_logger().warn(f'{self.cur}: force {self.force_f:.2f} N at the container, '
                                   f'ore may have been lost on the way')
            self.results[self.cur] = 'lost while carrying'
        else:
            self.get_logger().info(f'{self.cur}: still held at the container ({self.force_f:.2f} N)')
        self.magnet(False, lambda: self.pause(RELEASE_WAIT_S, self.after_release))

    def after_release(self):
        if self.force_f <= FREE_MAX_N:
            self.get_logger().info(f'{self.cur}: DROPPED (force {self.force_f:.2f} N)')
            if self.results.get(self.cur) == 'picked':
                self.results[self.cur] = 'picked and dropped'
        else:
            self.get_logger().warn(f'{self.cur}: force still {self.force_f:.2f} N after release')
        # Leave in JOINT space (as Task 1B did): a Cartesian lift at the far container
        # drives the elbow straight (singularity). SELECT starts the unfold move.
        self.ore_finished()

    def ore_finished(self):
        self.cur = None
        self.state = 'SELECT'

    def abort_ore(self, reason):
        self.get_logger().error(f'{self.cur}: aborted, {reason}')
        self.results[self.cur] = f'aborted: {reason}'
        self.pub_twist(0.0, 0.0, 0.0)

        def lift_away():
            p = self.tcp_xyz()
            self.start_tcp_path(
                [self.tgt(p[0], p[1], p[2] + 0.10, 0.02, LIFT_MPS)], self.ore_finished)

        self.magnet(False, lambda: self.pause(RELEASE_WAIT_S, lift_away))

    # ------------------------------------------------------------ magnet / pause states
    def do_magnet(self):
        if self.magnet_future is None:
            if not self.magnet_cli.service_is_ready():
                return
            req = SetBool.Request()
            req.data = self.magnet_target
            self.get_logger().info(f'Magnet {"ON" if self.magnet_target else "OFF"}')
            self.magnet_future = self.magnet_cli.call_async(req)
            return
        if not self.magnet_future.done():
            return
        try:
            res = self.magnet_future.result()
            if not res.success:
                self.get_logger().warn(f'Magnet service said: {res.message}')
        except Exception as e:
            self.get_logger().error(f'Magnet call failed: {e}')
        self.magnet_future = None
        fn, self.magnet_then = self.magnet_then, None
        fn()

    def do_pause(self):
        self.pub_twist(0.0, 0.0, 0.0)
        if self.now_s() >= self.pause_until:
            fn, self.pause_then = self.pause_then, None
            fn()

    # ------------------------------------------------------------ main timer
    def process(self):
        if (self.tcp_pose is None or self.joint_angles is None
                or self.arm_status is None or self.force is None):
            return
        s = self.state
        if s in ('LATCHED', 'DONE'):
            self.pub_twist(0.0, 0.0, 0.0)
        elif s == 'WAIT':
            self.do_wait()
        elif s == 'SCAN':
            self.do_scan()
        elif s == 'SELECT':
            self.do_select()
        elif s == 'JOINT':
            self.do_joint()
        elif s == 'SWING':
            self.do_swing()
        elif s == 'TCP':
            self.do_tcp()
        elif s == 'ORIENT':
            self.do_orient()
        elif s == 'MAGNET':
            self.do_magnet()
        elif s == 'PAUSE':
            self.do_pause()


def main():
    rclpy.init(args=sys.argv)
    node = pick_and_place()
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()