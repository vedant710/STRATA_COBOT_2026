#!/usr/bin/env python3


import sys
import math
import itertools

import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from scipy.spatial.transform import Rotation

from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32, Float32, String
from std_srvs.srv import SetBool

# ---------------------------------------------------------------- constants
ore_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']
PACKAGE_NAME = 'ore_package'

ready_topic = '/handover/ebot_ready'
taken_topic = '/handover/package_taken'

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
# CONTAINERS: TCP release points in base_link (Task 1B waypoints 3, 4, 5)
# ------------------------------------------------------------------------
CONTAINER_DROP = {
    'azurite_ore':    (-0.0763, 0.5280, 0.1967),
    'malachite_ore':  (-0.4097, 0.5280, 0.1967),
    'vanadinite_ore': (-0.7430, 0.5280, 0.1967),
}
DROP_TOL = {'azurite_ore': 0.020, 'malachite_ore': 0.020, 'vanadinite_ore': 0.025}

# CRUSHER: TCP release point in base_link, from the Task 1B waypoint 2 (ore crusher drop pose).
HOPPER_DROP = (-0.8000, -0.0005, 0.3967)
HOPPER_TOL = 0.020

# ---------------------------------------------------------------- tuning
START_DELAY_S = 0.2
EXPECTED_ORES = 4
SCAN_SETTLE_S = 0.4
SCAN_TIMEOUT_S = 25.0

# Base Cartesian speeds (m/s). The `speed_scale` parameter multiplies TRANSIT and LIFT only.
TRANSIT_MPS = 0.12             # x speed_scale 2.0 = 0.24 m/s (input cap 0.25)
PRECONTACT_MPS = 0.08
DESCEND_MPS = 0.015
LIFT_MPS = 0.06
SLOW_MPS = 0.03
MIN_MPS = 0.010
KP_LIN = 2.5
TARGET_TIMEOUT_S = 35.0

# Task 3 limits (never exceeded, whatever speed_scale is)
CART_CAP_MPS = 0.25
JOINT_CAP = 0.595             # hard input cap is 0.60 (rejected above it)

HOVER_TOL = 0.015
DESCEND_TOL = 0.002
MAX_ATTEMPTS = 3
PRESS_ABORT_N = 3.0

# Per-item parameters. 'ore' = as in Task 2A. 'package' = heavier (0.30 kg, ~2.94 N held).
# reach_margin: how far (m) the wrist must stay inside full arm stretch at a drop.
# lean_max_deg: largest outward tool lean the planner may use for that item.
ORE_PARAMS = {
    'kind': 'ore',
    'hover_dz': 0.09, 'precontact_dz': 0.022, 'lift_dz': 0.12, 'xy_tol': 0.005,
    'contact_dz': [0.001, -0.002, -0.004],
    'contact_n': [0.6, 0.9, 1.4],
    'press_abort': 3.0,
    'hold_min': 0.9, 'hold_max': 2.2,
    'lost_n': 0.5,
    'approach_dz': 0.05,
    'reach_margin': 0.035, 'lean_max_deg': 35.0,
}
PACKAGE_PARAMS = {
    'kind': 'package',
    'hover_dz': 0.12, 'precontact_dz': 0.025, 'lift_dz': 0.15, 'xy_tol': 0.008,
    'contact_dz': [0.001, -0.002, -0.004],
    'contact_n': [1.0, 1.0, 1.4],
    'press_abort': 3.5,
    'hold_min': 2.0, 'hold_max': 3.8,
    'lost_n': 1.5,
    'approach_dz': 0.10,
    'reach_margin': 0.015, 'lean_max_deg': 12.0,
    'lift_max_mps': 0.09,
}
PACKAGE_MAX_ABORTS = 3         # give up on the package after this many failed tries
PACKAGE_WAIT_TIMEOUT_S = 240.0

# Reach guard for item pick-up positions (horizontal distance from the base axis).
MAX_REACH_M = 0.95
ITEM_Z_RANGE = (-0.30, 0.60)   # loose sanity band for an item z in base_link

# Watchdog: one item (all attempts, carry and drop included) must finish within this time.
ITEM_TIMEOUT_S = 240.0

# Force readings (tool weight already subtracted)
FREE_MAX_N = 0.4
FORCE_ALPHA = 0.3
RELEASE_RETRIES = 2

# Tool orientation
TOOL_AXIS_TARGET = (0.0, 0.0, -1.0)
TILT_OK = 0.035
TILT_PROCEED = 0.10
ORIENT_HOLD_S = 0.10
ORIENT_TIMEOUT_S = 25.0
KP_ANG = 1.5
ANG_MAX = 0.45                 # Task 3 cap 0.60

# Timing
MAGNET_SETTLE_S = 0.08
CONTACT_SETTLE_S = 0.25
HOLD_CHECK_S = 0.35
RELEASE_WAIT_S = 0.35

# Joint space (base values, scaled by speed_scale, capped at JOINT_CAP)
JOINT_MAX = 0.35
JOINT_MIN = 0.06
KP_JOINT = 2.5
UNFOLD_TOL = 0.09
HOME_TOL = 0.02

# Direct transit between items (skips the detour through pan = 0)
DIRECT_TRANSIT = True
PAN_LIMIT = 5.8                # joint limit is +-6.28 rad; stay clear of it

# Overlap the retract with the pan swing. The pan is released gradually once the largest error
# of shoulder_lift / elbow / wrist_1 falls below FOLD_GATE_START (full speed below FOLD_GATE_FULL).
# Set OVERLAP_SWING = False for the old retract-then-swing sequence (safer near the crusher).
OVERLAP_SWING = True
FOLD_GATE_START = 0.35         # rad
FOLD_GATE_FULL = 0.15          # rad
UNFOLD_AZ_GUESS = math.radians(-22.4)   # TCP azimuth of the unfold pose at pan 0 (from your log)

# Contact is only believed this close above the item's target height (m)
CONTACT_GATE_DZ = 0.015

# After the last drop: True = unfold and return to the start pose (as in the 139 s run).
# False = stop where the arm is. Try False once and check in summary.txt that B is still earned.
RETURN_HOME_AT_END = True

# Pan swing
SAFE_RADIUS = 0.35
SWING_TOL = 0.08
SWING_MIN = 0.15
SWING_MAX = 0.595             # Task 3 cap 0.60
KP_SWING = 2.0

# ---- UR7e reach model, used to plan drops that never reach full stretch
UR_SHOULDER_Z = 0.1626
UR_A2, UR_A3 = 0.425, 0.3922
UR_L = UR_A2 + UR_A3            # wrist-1 distance from the shoulder at full stretch
UR_D4 = 0.1333                  # tool axis lateral offset from the arm plane
UR_D5 = 0.0997
UR_TOOL_LEN = 0.0996 + 0.0204   # wrist-3 -> flange + flange -> tcp

DROP_LEAN_STEP = math.radians(2.5)
DROP_LEAN_HARD_MAX = math.radians(40.0)
DROP_RETRIES = 2
DROP_RETRY_LEAN = math.radians(5.0)
GUARD_HARD_M = 0.010            # elbow this close to straight -> stop
GUARD_SOFT_M = 0.020            # ... or this close while status is 10/11
GUARD_RELEASE_XY = 0.05         # stopped this near the drop point -> release from here
GUARD_RELEASE_DZ = 0.09
AXIS_OK_DROP = 0.07             # rad, lean must be reached before waypoints count as done


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


def stretch_margin(x, y, z, lean):
    """Metres the wrist stays inside full arm stretch with the TCP at (x, y, z) and the tool
    leaning outward by `lean` rad. Negative = unreachable."""
    r = math.hypot(x, y)
    fwd = math.sqrt(max(r * r - UR_D4 * UR_D4, 0.0))
    tool_fwd = UR_D5 * math.cos(lean) + UR_TOOL_LEN * math.sin(lean)
    tool_up = UR_D5 * math.sin(lean) - UR_TOOL_LEN * math.cos(lean)
    u = fwd - tool_fwd
    h = (z - tool_up) - UR_SHOULDER_Z
    return UR_L - math.hypot(u, h)


def elbow_margin(q3):
    """Metres of wrist-to-shoulder distance left before the arm is fully straight."""
    d2 = UR_A2 ** 2 + UR_A3 ** 2 + 2.0 * UR_A2 * UR_A3 * math.cos(q3)
    return UR_L - math.sqrt(max(d2, 0.0))


class pick_and_place(Node):

    def __init__(self):
        super().__init__(
            'task3a_manipulation_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        # ---- speed scaling (free-space moves only)
        self.declare_parameter('speed_scale', 2.0)
        scale = float(self.get_parameter('speed_scale').value)
        scale = max(0.3, min(2.0, scale))
        self.speed_scale = scale
        self.transit_mps = min(TRANSIT_MPS * scale, CART_CAP_MPS)
        self.lift_mps = min(LIFT_MPS * scale, CART_CAP_MPS)
        self.joint_max = min(JOINT_MAX * scale, JOINT_CAP)
        self.get_logger().info(
            f'speed_scale={scale:.2f}: transit={self.transit_mps:.3f} m/s, '
            f'lift={self.lift_mps:.3f} m/s, joint={self.joint_max:.2f} rad/s')

        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)
        self.status_pub = self.create_publisher(String, '/ore_status', 10)
        self.taken_pub = self.create_publisher(Bool, taken_topic, latched)

        self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.create_subscription(JointState, '/joint_states', self.jointstatecb, 50)
        self.create_subscription(Int32, '/arm_status', self.armstatuscb, 10)
        self.create_subscription(
            Float32, '/ur7e/end_effector/force_magnitude', self.forcecb, 50)
        self.create_subscription(Bool, ready_topic, self.readycb, latched)

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
        self.force_f = None
        self.home = None
        self.last_missing_log_t = 0.0
        self.last_magnet_warn_t = 0.0

        # handover state
        self.ebot_ready = False
        self.package_done = False          # package dropped in the hopper
        self.package_gave_up = False
        self.package_aborts = 0
        self.wait_pkg_t0 = None

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

        # current item
        self.cur = None
        self.prm = ORE_PARAMS
        self.cur_pos = None
        self.attempt = 0
        self.picked = []
        self.results = {}
        self.drop = None
        self.drop_tol = 0.02
        self.item_t0 = None
        self.release_tries = 0
        self.run_t0 = None             # sim time when the arm first started an item
        self.unfold_az_offset = None   # TCP azimuth minus pan, measured in the unfold pose

        # tool axis target and drop planning
        self.axis_target = np.array([0.0, 0.0, -1.0])
        self.in_drop = False
        self.drop_tries = 0
        self.lean_bonus = 0.0

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

        self.publish_taken(False)           # eBot must not act on a stale True

        self.timer = self.create_timer(0.05, self.process)
        self.status_timer = self.create_timer(0.5, self.publish_status)
        self.get_logger().info('Task 3A manipulation node started')

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

    def readycb(self, msg):
        if msg.data and not self.ebot_ready:
            t = '' if self.run_t0 is None else f' (arm t={self.now_s() - self.run_t0:.1f} s)'
            self.get_logger().info(f'eBot is beside the arm (ebot_ready){t}')
        self.ebot_ready = bool(msg.data)

    def armstatuscb(self, data):
        self.arm_status = data.data
        if data.data != self.last_status:
            self.last_status = data.data
            self.get_logger().info(f'Arm status: {data.data}')
        if 20 <= data.data <= 23 and self.state != 'LATCHED':
            self.get_logger().error(
                f'PROTECTIVE STOP {data.data} during {self.state} (item {self.cur}). '
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

    def publish_taken(self, value):
        msg = Bool()
        msg.data = bool(value)
        self.taken_pub.publish(msg)

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

    def pos_ok(self, p):
        """Reach and height sanity check for an item position (base_link)."""
        if math.hypot(p[0], p[1]) > MAX_REACH_M:
            return False
        return ITEM_Z_RANGE[0] <= p[2] <= ITEM_Z_RANGE[1]

    def is_package(self):
        return self.cur == PACKAGE_NAME

    def package_wanted(self):
        return not self.package_done and not self.package_gave_up

    def package_ready(self):
        if not (self.package_wanted() and self.ebot_ready):
            return False
        p = self.ore_pos(PACKAGE_NAME)
        return p is not None and self.pos_ok(p)

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

    def axis_err(self):
        """Angle (rad) between the tool z axis and the current axis target."""
        b = self.axis_target / np.linalg.norm(self.axis_target)
        return math.acos(max(-1.0, min(1.0, float(np.dot(self.tool_axis(), b)))))

    def reset_axis(self):
        self.axis_target = np.array([0.0, 0.0, -1.0])
        self.in_drop = False
        self.drop_tries = 0
        self.lean_bonus = 0.0

    def angular_cmd(self):
        if TOOL_AXIS_TARGET is None:
            return 0.0, 0.0, 0.0
        a = self.tool_axis()
        b = self.axis_target.copy()
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
    def unfold_toward(self, x, y):
        """Joint queue that retracts the arm in place, then swings it in the compact (unfolded)
        pose so the TCP faces (x, y), taking the shortest way round. Falls back to the plain
        unfold at pan 0 until the unfold-pose azimuth offset has been measured."""
        plain = [('unfold', UNFOLDED, UNFOLD_TOL)]
        if (not DIRECT_TRANSIT or self.unfold_az_offset is None
                or self.joint_angles is None):
            return plain
        cur_pan = self.joint_angles[0]
        want = wrap(math.atan2(y, x) - self.unfold_az_offset)
        best = None
        for k in (-1, 0, 1):
            cand = want + 2.0 * math.pi * k
            if abs(cand) > PAN_LIMIT:
                continue
            if best is None or abs(cand - cur_pan) < abs(best - cur_pan):
                best = cand
        if best is None:
            return plain
        final = list(UNFOLDED)
        final[0] = best
        self.get_logger().info(
            f'Direct transit: pan {math.degrees(cur_pan):.0f} -> {math.degrees(best):.0f} deg '
            f'({"overlapped fold and swing" if OVERLAP_SWING else "retract first, then swing"})')
        if OVERLAP_SWING:
            return [('unfold-swing', final, UNFOLD_TOL)]
        retract = list(UNFOLDED)
        retract[0] = cur_pan
        return [('retract', retract, UNFOLD_TOL), ('unfold-swing', final, UNFOLD_TOL)]

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
            if label == 'unfold' and self.unfold_az_offset is None:
                tp = self.tcp_pose.position
                self.unfold_az_offset = wrap(math.atan2(tp.y, tp.x) - self.joint_angles[0])
                self.get_logger().info(
                    f'Unfold pose azimuth offset = {math.degrees(self.unfold_az_offset):.1f} deg')
            self.joint_queue.pop(0)
            if not self.joint_queue:
                self.joint_done()
            return
        vels = [clamp(KP_JOINT * e, self.joint_max) for e in errs]
        peak = max(abs(v) for v in vels)
        if 1e-9 < peak < JOINT_MIN:
            vels = [v * JOINT_MIN / peak for v in vels]
        if OVERLAP_SWING and label == 'unfold-swing':
            e_fold = max(abs(errs[i]) for i in (1, 2, 3))
            g = (FOLD_GATE_START - e_fold) / (FOLD_GATE_START - FOLD_GATE_FULL)
            vels[0] *= max(0.0, min(1.0, g))
        self.pub_joints(vels)

    def joint_done(self):
        if self.pending == 'SELECT':
            self.state = 'SELECT'
        elif self.pending == 'PICK':
            self.begin_pick()
        elif self.pending == 'CARRY':
            self.begin_drop_move()
        elif self.pending == 'WAITPKG':
            self.wait_pkg_t0 = self.now_s()
            self.get_logger().info('Ores done, waiting for the ore package')
            self.state = 'WAITPKG'
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
        mag = max(SWING_MIN, min(SWING_MAX, KP_SWING * abs(err)))
        self.pub_joints([math.copysign(mag, err) * self.pan_sign, 0, 0, 0, 0, 0])

    def drop_guard_trip(self, p, m):
        """The elbow is about to lock straight. Stop now, before status 21 latches."""
        self.pub_twist(0.0, 0.0, 0.0)
        d = self.drop
        near = (math.hypot(p[0] - d[0], p[1] - d[1]) <= GUARD_RELEASE_XY
                and p[2] - d[2] <= GUARD_RELEASE_DZ)
        if near:
            self.get_logger().warn(
                f'Elbow margin {m * 1000:.0f} mm: releasing from here, '
                f'{(p[2] - d[2]) * 1000:.0f} mm above the drop point')
            self.tcp_targets = []
            self.tcp_finished()
        elif self.drop_tries < DROP_RETRIES:
            self.drop_tries += 1
            self.lean_bonus += DROP_RETRY_LEAN
            self.get_logger().warn(
                f'Elbow margin {m * 1000:.0f} mm too small: retreating in joint space, '
                f'retry {self.drop_tries}/{DROP_RETRIES} with '
                f'{math.degrees(self.lean_bonus):.0f} deg more lean')
            self.tcp_targets = []
            self.start_joint_path(self.unfold_toward(d[0], d[1]), 'CARRY')
        else:
            self.abort_ore('drop pose too close to full stretch')

    def do_tcp(self):
        if not self.ensure_controller(twist_controller):
            return
        t = self.tcp_targets[0]
        tx, ty, tz = t['pos']
        p = self.tcp_xyz()
        ex, ey, ez = tx - p[0], ty - p[1], tz - p[2]
        dist = math.sqrt(ex * ex + ey * ey + ez * ez)
        now = self.now_s()
        force = self.force_f if self.force_f is not None else 0.0

        if now - self.last_log_t > 1.0:
            self.last_log_t = now
            self.get_logger().info(
                f'[{self.cur}] {t["kind"]}: dist={dist * 1000:.0f} mm, z={p[2]:.3f}, '
                f'force={force:.2f} N, tilt={math.degrees(self.tilt()):.1f} deg, '
                f'stretch={elbow_margin(self.joint_angles[2]) * 1000:.0f} mm, '
                f'status={self.arm_status}')

        # stop BEFORE the elbow locks straight (status 21 would latch)
        if self.in_drop and t['kind'] in ('standoff', 'drop'):
            m = elbow_margin(self.joint_angles[2])
            if m < GUARD_HARD_M or (self.arm_status in (10, 11) and m < GUARD_SOFT_M):
                self.drop_guard_trip(p, m)
                return

        # contact handling while descending onto the item
        if t['kind'] == 'descend':
            k = min(self.attempt, len(self.prm['contact_n']) - 1)
            if force >= self.prm['press_abort']:
                self.get_logger().warn(f'Pressing too hard ({force:.2f} N), stopping descent')
                self.tcp_targets = []
                self.tcp_finished()
                return
            if force >= self.prm['contact_n'][k] and p[2] <= tz + CONTACT_GATE_DZ:
                self.get_logger().info(f'Contact at {force:.2f} N, z={p[2]:.4f}')
                self.tcp_targets = []
                self.tcp_finished()
                return

        tol = t['tol']
        leaning = t['kind'] in ('standoff', 'drop')
        if dist <= tol and (not leaning or self.axis_err() < AXIS_OK_DROP):
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
                f'CONTAINER_DROP not filled in for {missing}. Not moving.')
            self.state = 'DONE'
            return
        if HOPPER_DROP is None:
            self.get_logger().warn(
                'HOPPER_DROP is not set: TEST MODE. Ores are sorted first, then the package is '
                'picked, lifted, /handover/package_taken is sent, and the package is HELD at the '
                'lift height (no drop).')
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
        for n in list(found):
            if not self.pos_ok(found[n]):
                if n not in self.results:
                    self.get_logger().error(
                        f'{n} at ({found[n][0]:.2f}, {found[n][1]:.2f}, {found[n][2]:.2f}) is '
                        f'outside the arm reach: ignored (false detection?)')
                    self.results[n] = 'out of reach'
                del found[n]
        changed = (set(found) != set(self.scan_prev) or
                   any(np.linalg.norm(found[n] - self.scan_prev[n]) > 0.005 for n in found))
        if changed or self.scan_stable_t is None:
            self.scan_stable_t = now
        self.scan_prev = found
        stable = now - self.scan_stable_t >= SCAN_SETTLE_S
        timed_out = now - self.scan_t0 > SCAN_TIMEOUT_S
        if stable and found and (len(found) >= EXPECTED_ORES or timed_out):
            order = self.plan_order(found)
            self.queue = order
            self.get_logger().info(f'Scan done, {len(order)} ores: {order}')
            for n in order:
                p = found[n]
                self.get_logger().info(f'  {n}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})')
            self.state = 'SELECT'
        elif timed_out and not found:
            self.get_logger().error('No ore transforms received. Is task3A_perception.py running?')
            self.scan_t0 = now

    def plan_order(self, found):
        """Order of the ores that minimises the total pan travel: pick -> container -> next pick."""
        names = list(found)
        if len(names) > 6:
            ref = self.tcp_xyz()
            return sorted(names, key=lambda n: np.linalg.norm(found[n][:2] - ref[:2]))
        az = lambda p: math.atan2(p[1], p[0])
        best, best_c = names, float('inf')
        for perm in itertools.permutations(names):
            cur, c = UNFOLD_AZ_GUESS, 0.0
            for n in perm:
                a = az(found[n])
                d = CONTAINER_DROP[n.rsplit('_', 1)[0]]
                ca = math.atan2(d[1], d[0])
                c += abs(wrap(a - cur)) + abs(wrap(ca - a))
                cur = ca
            if c < best_c:
                best, best_c = list(perm), c
        self.get_logger().info(f'Ore order by pan travel: {best} ({math.degrees(best_c):.0f} deg)')
        return best

    def start_item(self, name, params, label):
        self.reset_axis()
        self.cur = name
        self.prm = params
        self.item_t0 = self.now_s()
        if self.run_t0 is None:
            self.run_t0 = self.item_t0
        self.release_tries = 0
        self.get_logger().info(
            f'===== Processing {name}{label} (arm t={self.item_t0 - self.run_t0:.1f} s) =====')
        pos = self.ore_pos(name)
        queue = (self.unfold_toward(pos[0], pos[1]) if pos is not None
                 else [('unfold', UNFOLDED, UNFOLD_TOL)])
        self.start_joint_path(queue, 'PICK')

    def do_select(self):
        # 1) the package goes first as soon as the eBot is waiting with it
        # without HOPPER_DROP (test mode) the ores come first, so the magnet is free for them
        if self.package_ready() and (HOPPER_DROP is not None or not self.queue):
            self.start_item(PACKAGE_NAME, PACKAGE_PARAMS, ' (handover)')
            return
        # 2) next ore
        if self.queue:
            self.start_item(self.queue.pop(0), ORE_PARAMS, '')
            return
        self.cur = None
        self.item_t0 = None
        # 3) only the package is left and it has not arrived: wait at the unfold pose
        if self.package_wanted():
            self.start_joint_path([('unfold', UNFOLDED, UNFOLD_TOL)], 'WAITPKG')
            return
        # 4) all done
        if not RETURN_HOME_AT_END:
            self.finish_run()
            return
        self.start_joint_path([('unfold', UNFOLDED, UNFOLD_TOL),
                               ('home', self.home, HOME_TOL)], 'FINISH')

    def do_waitpkg(self):
        self.pub_twist(0.0, 0.0, 0.0)
        now = self.now_s()
        if now - self.last_log_t > 5.0:
            self.last_log_t = now
            have = self.ore_pos(PACKAGE_NAME) is not None
            self.get_logger().info(
                f'Waiting for the package: ebot_ready={self.ebot_ready}, '
                f'ore_package transform={"yes" if have else "NO (is perception running?)"}')
        if self.package_ready():
            self.state = 'SELECT'
            return
        if self.now_s() - self.wait_pkg_t0 > PACKAGE_WAIT_TIMEOUT_S:
            self.get_logger().error('Gave up waiting for the ore package')
            self.package_gave_up = True
            self.state = 'SELECT'

    def finish_run(self):
        t = '' if self.run_t0 is None else f' in {self.now_s() - self.run_t0:.1f} s of arm time'
        self.get_logger().info(f'RUN COMPLETE{t}. Results: {self.results}')
        self.state = 'DONE'

    # ---- pick
    def begin_pick(self):
        p = self.ore_pos(self.cur)
        if p is None or not self.pos_ok(p):
            reason = 'no transform' if p is None else 'position out of range'
            self.results[self.cur] = reason
            self.get_logger().error(f'{self.cur}: {reason}, skipping')
            if self.is_package():
                self.package_aborts += 1
                if self.package_aborts >= PACKAGE_MAX_ABORTS:
                    self.package_gave_up = True
            self.cur = None
            self.item_t0 = None
            self.state = 'SELECT'
            return
        self.cur_pos = p
        self.attempt = 0
        self.start_attempt()

    def start_attempt(self):
        p = self.ore_pos(self.cur)
        if p is not None and self.pos_ok(p):
            self.cur_pos = p
        x, y, z = self.cur_pos
        self.hover_pos = (x, y, z + self.prm['hover_dz'])
        self.get_logger().info(
            f'{self.cur}: attempt {self.attempt + 1}/{MAX_ATTEMPTS}, item at '
            f'({x:.3f}, {y:.3f}, {z:.3f})')
        self.start_tcp_path(
            [self.tgt(x, y, z + self.prm['hover_dz'], HOVER_TOL, self.transit_mps)],
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
            elif now - self.orient_ok_t >= ORIENT_HOLD_S or TOOL_AXIS_TARGET is None:
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
        if p is not None and self.pos_ok(p):
            self.cur_pos = p
        x, y, z = self.cur_pos
        k = min(self.attempt, len(self.prm['contact_dz']) - 1)
        pre = self.prm['precontact_dz']
        cdz = self.prm['contact_dz'][k]
        xy_tol = self.prm['xy_tol']
        self.pause(MAGNET_SETTLE_S, lambda: self.start_tcp_path([
            self.tgt(x, y, z + pre, xy_tol, PRECONTACT_MPS),
            self.tgt(x, y, z + cdz, DESCEND_TOL, DESCEND_MPS, 'descend'),
        ], self.after_descend))

    def after_descend(self):
        self.pause(CONTACT_SETTLE_S, self.start_lift)

    def start_lift(self):
        x, y, z = self.cur_pos
        speed = min(self.lift_mps, self.prm.get('lift_max_mps', CART_CAP_MPS))
        self.start_tcp_path(
            [self.tgt(x, y, z + self.prm['lift_dz'], 0.01, speed, 'lift')],
            self.after_lift)

    def after_lift(self):
        self.pause(HOLD_CHECK_S, self.check_hold)

    def check_hold(self):
        f = self.force_f if self.force_f is not None else 0.0
        if self.prm['hold_min'] <= f <= self.prm['hold_max']:
            self.get_logger().info(f'{self.cur}: PICKED (force {f:.2f} N)')
            self.picked.append(self.cur)
            self.results[self.cur] = 'picked'
            if self.is_package():
                # the package is held and clear of the deck: the eBot may leave now
                self.publish_taken(True)
                self.get_logger().info(
                    f'/handover/package_taken = True (arm t={self.now_s() - self.run_t0:.1f} s)')
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
        """Unfold first, swing from there (same safe sequence as Task 1B, reversed)."""
        if self.is_package() and HOPPER_DROP is None:
            self.get_logger().warn(
                f'{self.cur}: PICKED and LIFTED, package_taken sent. HOPPER_DROP is not set, '
                f'so the package is held here and the run stops (test mode).')
            self.results[self.cur] = 'held (no HOPPER_DROP)'
            self.package_done = True
            self.pub_twist(0.0, 0.0, 0.0)
            self.state = 'DONE'
            return
        if self.is_package():
            self.drop = HOPPER_DROP
            self.drop_tol = HOPPER_TOL
        else:
            ot = self.cur.rsplit('_', 1)[0]
            self.drop = CONTAINER_DROP[ot]
            self.drop_tol = DROP_TOL[ot]
        self.get_logger().info(f'{self.cur}: carrying, going to the unfold pose before the swing')
        self.start_joint_path(self.unfold_toward(self.drop[0], self.drop[1]), 'CARRY')

    def plan_drop_lean(self, cx, cy, cz):
        """Smallest outward lean that keeps the wrist `reach_margin` inside full stretch
        at BOTH the stand-off point and the drop point."""
        zs = (cz + self.prm['approach_dz'], cz)
        need = self.prm['reach_margin']
        lean_max = math.radians(self.prm['lean_max_deg'])
        a = 0.0
        while True:
            m = min(stretch_margin(cx, cy, z, a) for z in zs)
            if m >= need or a >= lean_max - 1e-9:
                return a, m
            a = min(a + DROP_LEAN_STEP, lean_max)

    def begin_drop_move(self):
        cx, cy, cz = self.drop
        zs = (cz + self.prm['approach_dz'], cz)
        lean, _ = self.plan_drop_lean(cx, cy, cz)
        lean = min(lean + self.lean_bonus, DROP_LEAN_HARD_MAX)
        margin = min(stretch_margin(cx, cy, z, lean) for z in zs)
        if margin < 0.010:
            self.in_drop = True        # so the abort leaves in joint space
            self.abort_ore(f'drop point unreachable (margin {margin * 1000:.0f} mm at '
                           f'{math.degrees(lean):.0f} deg lean). Check the container coordinates')
            return
        r = math.hypot(cx, cy)
        self.axis_target = np.array([math.sin(lean) * cx / r, math.sin(lean) * cy / r,
                                     -math.cos(lean)])
        self.in_drop = True
        self.get_logger().info(
            f'{self.cur}: drop at reach {r:.3f} m -> tool lean {math.degrees(lean):.1f} deg '
            f'outward, planned wrist margin {margin * 1000:.0f} mm')
        self.start_tcp_path(
            [self.tgt(cx, cy, cz + self.prm['approach_dz'], 0.03, self.transit_mps, 'standoff'),
             self.tgt(cx, cy, cz, self.drop_tol, 0.10 if r < 0.75 else PRECONTACT_MPS, 'drop')],
            self.at_drop, swing_to=(cx, cy))

    def at_drop(self):
        self.pause(HOLD_CHECK_S, self.release_now)

    def release_now(self):
        f = self.force_f if self.force_f is not None else 0.0
        if f < self.prm['lost_n']:
            self.get_logger().warn(f'{self.cur}: force {f:.2f} N at the drop, '
                                   f'item may have been lost on the way')
            self.results[self.cur] = 'lost while carrying'
        else:
            self.get_logger().info(f'{self.cur}: still held at the drop ({f:.2f} N)')
        self.magnet(False, lambda: self.pause(RELEASE_WAIT_S, self.after_release))

    def after_release(self):
        f = self.force_f if self.force_f is not None else 0.0
        if f <= FREE_MAX_N:
            self.get_logger().info(f'{self.cur}: DROPPED (force {f:.2f} N)')
            if self.results.get(self.cur) == 'picked':
                self.results[self.cur] = 'picked and dropped'
        elif self.release_tries < RELEASE_RETRIES:
            # item still pulling on the tool: switch the magnet off again and re-check
            self.release_tries += 1
            self.get_logger().warn(
                f'{self.cur}: force still {f:.2f} N after release, '
                f'retry {self.release_tries}/{RELEASE_RETRIES}')
            self.magnet(False, lambda: self.pause(RELEASE_WAIT_S, self.after_release))
            return
        else:
            self.get_logger().warn(f'{self.cur}: force still {f:.2f} N after release, moving on')
        if self.is_package():
            self.package_done = True
        # Leave in JOINT space: the next start_item retracts the arm before any Cartesian move.
        self.ore_finished()

    def ore_finished(self):
        if self.cur and self.item_t0 is not None and self.run_t0 is not None:
            now = self.now_s()
            self.get_logger().info(
                f'TIMING {self.cur}: item took {now - self.item_t0:.1f} s, '
                f'arm t={now - self.run_t0:.1f} s')
        self.reset_axis()
        self.cur = None
        self.item_t0 = None
        self.state = 'SELECT'

    def abort_ore(self, reason):
        self.get_logger().error(f'{self.cur}: aborted, {reason}')
        self.results[self.cur] = f'aborted: {reason}'
        self.item_t0 = None            # watchdog must not fire again during the abort
        self.pub_twist(0.0, 0.0, 0.0)
        if self.is_package():
            self.package_aborts += 1
            if self.package_aborts >= PACKAGE_MAX_ABORTS:
                self.get_logger().error('Giving up on the ore package')
                self.package_gave_up = True

        def lift_away():
            if self.in_drop:
                # a Cartesian lift near full stretch can latch; start_item retracts in joint space
                self.ore_finished()
                return
            p = self.tcp_xyz()
            self.start_tcp_path(
                [self.tgt(p[0], p[1], p[2] + 0.10, 0.02, self.lift_mps)], self.ore_finished)

        self.magnet(False, lambda: self.pause(RELEASE_WAIT_S, lift_away))

    # ------------------------------------------------------------ magnet / pause states
    def do_magnet(self):
        if self.magnet_future is None:
            if not self.magnet_cli.service_is_ready():
                now = self.now_s()
                if now - self.last_magnet_warn_t > 5.0:
                    self.last_magnet_warn_t = now
                    self.get_logger().warn('/magnet service not available yet, waiting...')
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
    def report_missing(self):
        now = self.now_s()
        if now - self.last_missing_log_t < 5.0:
            return
        self.last_missing_log_t = now
        missing = []
        if self.tcp_pose is None:
            missing.append('/tcp_pose_raw')
        if self.joint_angles is None:
            missing.append('/joint_states')
        if self.arm_status is None:
            missing.append('/arm_status')
        if self.force is None:
            missing.append('/ur7e/end_effector/force_magnitude')
        self.get_logger().warn(f'Waiting for sensor data: {", ".join(missing)}')

    def process(self):
        if (self.tcp_pose is None or self.joint_angles is None
                or self.arm_status is None or self.force is None):
            self.report_missing()
            return
        s = self.state

        # per-item watchdog
        if (self.item_t0 is not None and self.cur is not None
                and s not in ('LATCHED', 'DONE', 'MAGNET', 'PAUSE')
                and self.now_s() - self.item_t0 > ITEM_TIMEOUT_S):
            self.abort_ore(f'item watchdog ({ITEM_TIMEOUT_S:.0f} s)')
            return

        if s in ('LATCHED', 'DONE'):
            self.pub_twist(0.0, 0.0, 0.0)
        elif s == 'WAIT':
            self.do_wait()
        elif s == 'SCAN':
            self.do_scan()
        elif s == 'SELECT':
            self.do_select()
        elif s == 'WAITPKG':
            self.do_waitpkg()
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
                node.pub_joints([0.0] * 6)
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