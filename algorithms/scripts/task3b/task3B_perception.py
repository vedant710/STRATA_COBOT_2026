#!/usr/bin/env python3


import sys
import math

import cv2
import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, PointStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_geometry_msgs import do_transform_point

# ---------------------------------------------------------------- constants
ore_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']
PACKAGE_NAME = 'ore_package'
PACKAGE_TYPE = 'package'

color_topic = '/camera/camera/color/image_raw'
depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
camera_info_topic = '/camera/camera/color/camera_info'
ready_topic = '/handover/ebot_ready'
base_frame = 'base_link'

COLOR_RANGES = {
    'azurite_ore':    ([90, 120, 60],  [120, 255, 255]),
    'malachite_ore':  ([60, 120, 80],  [90, 255, 255]),
    'vanadinite_ore': ([0, 200, 100],  [20, 255, 255]),
}

# The ore package's top face is magenta (hue 154 in OpenCV's 0..179). No ore colour overlaps.
PACKAGE_HSV = ([140, 120, 60], [170, 255, 255])
PKG_MIN_AREA = 200         # px, the package top face is larger than an ore's
PKG_MAX_REACH = 1.0        # m, horizontal distance from the arm base axis

# ---------------------------------------------------------------- tuning
EXPECTED_ORES = 4
MAX_PER_TYPE = 2
MIN_AREA = 150
MIN_FILL = 0.60
MIN_ASPECT = 0.40
MATCH_DIST = 0.08
MERGE_DIST = 0.06
EMA_ALPHA = 0.40
DEPTH_PATCH = 4

# Ores lie on the conveyor (y ~ -0.65..-0.80, top face z ~ -0.064 in base_link, from the logs).
# The orange-red terrain has almost the vanadinite colour and was detected as an ore at
# (0.86, 1.28, -0.10) and (1.9, 1.2, -0.39): keep only the conveyor strip. If your ores
# are ever missed, widen this box rather than removing it.
ORE_REGION = (-1.0, 1.0, -1.2, -0.2, -0.12, -0.02)

# OPTIONAL box in base_link, (x_min, x_max, y_min, y_max, z_min, z_max), around where the
# package sits at the stop pose. None = no box: the package is found by its magenta colour
# alone (within PKG_MAX_REACH). Set it later to make detection stricter, or to enable the
# depth fallback when the colour is not seen.
PACKAGE_REGION = None

PKG_STRIDE = 4             # use every 4th pixel of the depth image
PKG_MIN_POINTS = 40        # fewer points in the box than this = no package
PKG_TOP_BAND = 0.015       # m, points within this of the highest ones form the top face
PKG_EMA_ALPHA = 0.30


def detect_ores(image):
    '''
    Returns:
        center_ore_list (list): (cX, cY) pixel of every ore-coloured face
        ore_type_list   (list): type of each, from ore_types
        area_list       (list): contour area in px (used to rank duplicates)
    '''
    center_ore_list, ore_type_list, area_list = [], [], []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)

    for ore_name, (lo, hi) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                continue

            (w, h) = cv2.minAreaRect(contour)[1]
            if w < 1.0 or h < 1.0:
                continue
            if area / (w * h) < MIN_FILL:
                continue
            if min(w, h) / max(w, h) < MIN_ASPECT:
                continue

            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            center_ore_list.append((int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])))
            ore_type_list.append(ore_name)
            area_list.append(area)

    return center_ore_list, ore_type_list, area_list


def detect_package(image):
    '''
    Returns:
        list of (cX, cY, area, (x, y, w, h)) for every large filled magenta blob, biggest first.
        Thin outlines (the magenta hopper outline) fail the fill test and are dropped.
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(PACKAGE_HSV[0]), np.array(PACKAGE_HSV[1]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    out = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < PKG_MIN_AREA:
            continue
        (w, h) = cv2.minAreaRect(contour)[1]
        if w < 1.0 or h < 1.0 or area / (w * h) < MIN_FILL:
            continue
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        out.append((int(M['m10'] / M['m00']), int(M['m01'] / M['m00']), area,
                    cv2.boundingRect(contour)))
    out.sort(key=lambda d: -d[2])
    return out


class ore_tf(Node):

    def __init__(self):
        super().__init__(
            'ore_tf_publisher',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self.create_subscription(Image, color_topic, self.colorimagecb, 10)
        self.create_subscription(Image, depth_topic, self.depthimagecb, 10)
        self.create_subscription(CameraInfo, camera_info_topic, self.caminfocb, 10)
        self.create_subscription(String, '/ore_status', self.orestatuscb, 10)
        self.create_subscription(Bool, ready_topic, self.readycb, latched)

        self.debug_pub = self.create_publisher(Image, '/ore_debug_image', 1)
        self.bridge = CvBridge()
        self.dbg_boxes = []
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.process_image)

        self.cv_image = None
        self.depth_image = None
        self.depth_frame = None
        self.cam_info = None
        self.fx = self.fy = self.cx = self.cy = 0.0

        # name -> {'type': str, 'pos': np.ndarray(3)}   (base_link, top-face centre)
        self.registry = {}
        self.picked = set()
        self.active = None
        self.locked = False
        self.ebot_ready = False
        self.warned_region = False

        self.get_logger().info('Task 3A perception node started')

    # ------------------------------------------------------------ callbacks
    def depthimagecb(self, data):
        try:
            depth = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            self.depth_frame = data.header.frame_id
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0
            else:
                depth = depth.astype(np.float32)
            self.depth_image = depth
        except CvBridgeError as e:
            self.get_logger().error(f'Error converting depth image: {e}')

    def colorimagecb(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Error converting colour image: {e}')

    def caminfocb(self, data):
        self.cam_info = data
        self.fx = float(data.k[0])
        self.fy = float(data.k[4])
        self.cx = float(data.k[2])
        self.cy = float(data.k[5])

    def readycb(self, msg):
        if msg.data and not self.ebot_ready:
            self.get_logger().info('eBot is beside the arm: looking for the ore package')
        self.ebot_ready = bool(msg.data)

    def orestatuscb(self, msg):
        # format: "locked=1;active=<name or ->;picked=a,b,c"
        try:
            fields = dict(f.split('=', 1) for f in msg.data.split(';') if '=' in f)
            self.locked = fields.get('locked', '0') == '1'
            act = fields.get('active', '-')
            self.active = None if act in ('', '-') else act
            picked = fields.get('picked', '')
            self.picked = set(n for n in picked.split(',') if n)
        except ValueError:
            pass

    # ------------------------------------------------------------ geometry
    def pixel_to_base(self, cX, cY):
        '''Pixel + aligned depth -> point in base_link (np.array) or None.'''
        height, width = self.depth_image.shape
        x1, x2 = max(0, cX - DEPTH_PATCH), min(width, cX + DEPTH_PATCH + 1)
        y1, y2 = max(0, cY - DEPTH_PATCH), min(height, cY + DEPTH_PATCH + 1)
        patch = self.depth_image[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None
        z = float(np.median(valid))

        pc = PointStamped()
        pc.header.stamp = self.get_clock().now().to_msg()
        pc.header.frame_id = self.depth_frame
        pc.point.x = (cX - self.cx) * z / self.fx
        pc.point.y = (cY - self.cy) * z / self.fy
        pc.point.z = z

        try:
            tf = self.tf_buffer.lookup_transform(base_frame, self.depth_frame, rclpy.time.Time())
            pb = do_transform_point(pc, tf)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        return np.array([pb.point.x, pb.point.y, pb.point.z])

    def in_box(self, p, box):
        x0, x1, y0, y1, z0, z1 = box
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and z0 <= p[2] <= z1

    def in_region(self, p):
        if ORE_REGION is not None and not self.in_box(p, ORE_REGION):
            return False
        if PACKAGE_REGION is not None and self.in_box(p, PACKAGE_REGION):
            return False                    # package area: never an ore
        return True

    def package_from_depth(self):
        '''Top-face centre of whatever stands inside PACKAGE_REGION, in base_link, or None.'''
        if PACKAGE_REGION is None:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(base_frame, self.depth_frame, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        q = tf.transform.rotation
        t = tf.transform.translation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T = np.array([t.x, t.y, t.z])

        h, w = self.depth_image.shape
        vs = np.arange(0, h, PKG_STRIDE)
        us = np.arange(0, w, PKG_STRIDE)
        uu, vv = np.meshgrid(us, vs)
        z = self.depth_image[vv, uu]
        ok = np.isfinite(z) & (z > 0.0)
        uu, vv, z = uu[ok], vv[ok], z[ok]
        if z.size == 0:
            return None
        cam = np.stack([(uu - self.cx) * z / self.fx, (vv - self.cy) * z / self.fy, z], axis=1)
        pb = cam @ R.T + T

        x0, x1, y0, y1, z0, z1 = PACKAGE_REGION
        m = ((pb[:, 0] >= x0) & (pb[:, 0] <= x1) & (pb[:, 1] >= y0) & (pb[:, 1] <= y1)
             & (pb[:, 2] >= z0) & (pb[:, 2] <= z1))
        sel = pb[m]
        if len(sel) < PKG_MIN_POINTS:
            return None
        z_top = float(np.percentile(sel[:, 2], 90))
        top = sel[sel[:, 2] >= z_top - PKG_TOP_BAND]
        if len(top) < max(5, PKG_MIN_POINTS // 3):
            return None
        return np.array([float(top[:, 0].mean()), float(top[:, 1].mean()),
                         float(np.median(top[:, 2]))])

    # ------------------------------------------------------------ tracking
    def n_ores(self):
        return sum(1 for e in self.registry.values() if e['type'] != PACKAGE_TYPE)

    def dedupe(self, detections):
        '''Keep at most MAX_PER_TYPE blobs per type, biggest first, no near-duplicates.'''
        kept = []
        for d in sorted(detections, key=lambda d: -d[2]):
            same = [k for k in kept if k[0] == d[0]]
            if len(same) >= MAX_PER_TYPE:
                continue
            if any(np.linalg.norm(k[1] - d[1]) < MERGE_DIST for k in same):
                continue
            kept.append(d)
        return kept

    def update_registry(self, detections):
        for ore_type in ore_types:
            dets = [d for d in detections if d[0] == ore_type]
            names = [n for n, e in self.registry.items()
                     if e['type'] == ore_type and n not in self.picked]

            pairs = []
            for di, d in enumerate(dets):
                for n in names:
                    dist = float(np.linalg.norm(d[1] - self.registry[n]['pos']))
                    if dist <= MATCH_DIST:
                        pairs.append((dist, di, n))
            pairs.sort(key=lambda x: x[0])

            used_d, used_n = set(), set()
            for dist, di, n in pairs:
                if di in used_d or n in used_n:
                    continue
                used_d.add(di)
                used_n.add(n)
                if n == self.active:
                    continue
                e = self.registry[n]
                e['pos'] = (1.0 - EMA_ALPHA) * e['pos'] + EMA_ALPHA * dets[di][1]

            for di, d in enumerate(dets):
                if di in used_d:
                    continue
                if self.locked or self.n_ores() >= EXPECTED_ORES:
                    continue
                free = [i for i in range(1, MAX_PER_TYPE + 1)
                        if f'{ore_type}_{i}' not in self.registry]
                if not free:
                    continue
                name = f'{ore_type}_{free[0]}'
                self.registry[name] = {'type': ore_type, 'pos': d[1].copy()}
                p = d[1]
                self.get_logger().info(
                    f'New ore {name} at base_link=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})')

    def package_from_colour(self):
        '''Top-face centre of the magenta package in base_link, or None.'''
        self.dbg_boxes = []
        for cX, cY, area, rect in detect_package(self.cv_image):
            p = self.pixel_to_base(cX, cY)
            if p is None or math.hypot(p[0], p[1]) > PKG_MAX_REACH:
                continue
            if PACKAGE_REGION is not None and not self.in_box(p, PACKAGE_REGION):
                continue
            self.dbg_boxes.append((rect, p))
            return p
        return None

    def update_package(self):
        '''Track the ore package while it is on the eBot beside the arm.'''
        if not self.ebot_ready or PACKAGE_NAME in self.picked:
            return
        if self.active == PACKAGE_NAME:
            return                          # arm is on it: keep the position frozen
        p = self.package_from_colour()
        if p is None and PACKAGE_REGION is not None:
            p = self.package_from_depth()   # fallback when the colour is not visible
        if p is None:
            return
        e = self.registry.get(PACKAGE_NAME)
        if e is None:
            self.registry[PACKAGE_NAME] = {'type': PACKAGE_TYPE, 'pos': p}
            self.get_logger().info(
                f'Package found at base_link=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})')
        else:
            e['pos'] = (1.0 - PKG_EMA_ALPHA) * e['pos'] + PKG_EMA_ALPHA * p

    def publish_debug(self, detections_px):
        '''Annotated colour image on /ore_debug_image (view with rqt_image_view).'''
        img = self.cv_image.copy()
        for (cX, cY), t, p in detections_px:
            cv2.circle(img, (cX, cY), 8, (0, 255, 0), 2)
            cv2.putText(img, f'{t} z={p[2]:.2f}', (cX + 10, cY), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)
        for (x, y, w, h), p in self.dbg_boxes:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img, f'ore_package ({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})', (x, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))

    def broadcast(self):
        stamp = self.get_clock().now().to_msg()
        msgs = []
        for name, e in self.registry.items():
            if name in self.picked:
                continue
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = base_frame
            t.child_frame_id = name
            t.transform.translation.x = float(e['pos'][0])
            t.transform.translation.y = float(e['pos'][1])
            t.transform.translation.z = float(e['pos'][2])
            t.transform.rotation.w = 1.0
            msgs.append(t)
        if msgs:
            self.br.sendTransform(msgs)

    # ------------------------------------------------------------ main timer
    def process_image(self):
        if self.cv_image is None or self.depth_image is None or self.cam_info is None:
            return
        if self.fx == 0.0 or self.fy == 0.0 or self.depth_frame is None:
            return

        centers, types, areas = detect_ores(self.cv_image)
        detections = []
        shown = []
        for (cX, cY), t, a in zip(centers, types, areas):
            p = self.pixel_to_base(cX, cY)
            if p is None or not self.in_region(p):
                continue
            detections.append((t, p, a))
            shown.append(((cX, cY), t, p))

        self.update_registry(self.dedupe(detections))
        self.update_package()
        self.broadcast()
        self.publish_debug(shown)


def main():
    rclpy.init(args=sys.argv)
    node = ore_tf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()