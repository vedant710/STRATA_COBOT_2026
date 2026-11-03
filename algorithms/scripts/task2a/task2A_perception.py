#!/usr/bin/env python3
'''
StrataCobot (SC) Theme (eYRC 2026-27) - Task 2A : PERCEPTION

Team ID:          [ Team-ID ]
Author List:      [ Names ]
Filename:         task2A_perception.py
Nodes:
    Publishing  - /tf  (one transform per ore: base_link -> <ore_type>_<id>)
    Subscribing - /camera/camera/color/image_raw
                  /camera/camera/aligned_depth_to_color/image_raw
                  /camera/camera/color/camera_info
                  /ore_status   (from task2A_manipulation.py)

What changed from Task 1A
-------------------------
1. Ore ids are tracked in base_link (3D), not in pixels. The camera view can change
   and the arm can occlude ores, so pixels are not a stable key.
2. The last known position of every ore still on the conveyor is re-broadcast on
   every tick, even while the arm hides it from the camera.
3. The manipulation node reports which ore it is picking and which are already
   picked. A picked ore stops being published (the task allows this) and a
   carried ore can never be re-detected as a "new" ore.
4. Candidate blobs are filtered by shape (the blue/green/orange outlines of the
   containers on the rover must not be mistaken for ores).
'''

import sys
import math

import cv2
import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, PointStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_geometry_msgs import do_transform_point

# ---------------------------------------------------------------- constants
ore_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']

color_topic = '/camera/camera/color/image_raw'
depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
camera_info_topic = '/camera/camera/color/camera_info'
base_frame = 'base_link'

# HSV ranges (same as Task 1A)
COLOR_RANGES = {
    'azurite_ore':    ([90, 120, 60],  [120, 255, 255]),   # blue
    'malachite_ore':  ([60, 120, 80],  [90, 255, 255]),    # green
    'vanadinite_ore': ([0, 200, 100],  [20, 255, 255]),    # orange
}

# ---------------------------------------------------------------- tuning
EXPECTED_ORES = 4          # four ores per run
MAX_PER_TYPE = 2           # ids are 1 or 2
MIN_AREA = 150             # px, noise floor
MIN_FILL = 0.60            # contour area / min-area-rect area (rejects thin outlines)
MIN_ASPECT = 0.40          # short side / long side of the min-area-rect
MATCH_DIST = 0.08          # m, same ore if it moved less than this between frames
MERGE_DIST = 0.06          # m, two blobs of one type closer than this are one ore
EMA_ALPHA = 0.40           # smoothing of the published position
DEPTH_PATCH = 4            # half-size (px) of the depth window around the centroid

# Optional workspace box in base_link to ignore anything outside the conveyor,
# (x_min, x_max, y_min, y_max, z_min, z_max). None = no filtering.
ORE_REGION = None


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

            # the ore's coloured face is a filled square; outlines and slivers are not
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


class ore_tf(Node):

    def __init__(self):
        super().__init__(
            'ore_tf_publisher',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        self.create_subscription(Image, color_topic, self.colorimagecb, 10)
        self.create_subscription(Image, depth_topic, self.depthimagecb, 10)
        self.create_subscription(CameraInfo, camera_info_topic, self.caminfocb, 10)
        self.create_subscription(String, '/ore_status', self.orestatuscb, 10)

        self.bridge = CvBridge()
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
        self.active = None          # ore the arm is currently picking (position frozen)
        self.locked = False         # no new ores may be created once picking begins

        self.get_logger().info('Task 2A perception node started')

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

    def in_region(self, p):
        if ORE_REGION is None:
            return True
        x0, x1, y0, y1, z0, z1 = ORE_REGION
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and z0 <= p[2] <= z1

    # ------------------------------------------------------------ tracking
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
                if n == self.active:        # arm is on it, the view is unreliable
                    continue
                e = self.registry[n]
                e['pos'] = (1.0 - EMA_ALPHA) * e['pos'] + EMA_ALPHA * dets[di][1]

            for di, d in enumerate(dets):
                if di in used_d:
                    continue
                if self.locked or len(self.registry) >= EXPECTED_ORES:
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
        for (cX, cY), t, a in zip(centers, types, areas):
            p = self.pixel_to_base(cX, cY)
            if p is None or not self.in_region(p):
                continue
            detections.append((t, p, a))

        self.update_registry(self.dedupe(detections))
        self.broadcast()


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