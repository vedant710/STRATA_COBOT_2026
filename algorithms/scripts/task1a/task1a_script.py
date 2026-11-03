#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1A of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:		    task1a_boilerplate.py
# Functions:
#			        [ Comma separated list of functions in this file ]
# Nodes:		    Add your publishing and subscribing node
#                   Example:
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/camera/color/image_raw, /etc... ]


################### IMPORT MODULES #######################

import rclpy
import sys
import cv2
import math
import tf2_ros
import numpy as np

from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, PointStamped
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point


##################### TASK CONSTANTS #######################

# Two ores of each type are spawned - six in all - told apart by an id of 1 or 2.
ore_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']

# The RealSense topics. The depth image is ALIGNED to the colour image.
color_topic = '/camera/camera/color/image_raw'
depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
camera_info_topic = '/camera/camera/color/camera_info'

# The parent frame every ore transform is published against.
base_frame = 'base_link'


##################### FUNCTION DEFINITIONS #######################

def detect_ores(image):
    '''
    Description:    Function to detect the ores present in a colour image frame and
                    return the pixel location and the type of each one found.

    Args:
        image                   (Image):    Input colour image frame received from the camera topic

    Returns:
        center_ore_list         (list):     Center pixel (cX, cY) of every ore detected in the frame
        ore_type_list           (list):     Type of each ore detected, taken from 'ore_types'
    '''

    ############ Function VARIABLES ############

    # ->  You can remove these variables if needed. These are just for suggestions to let you get started

    center_ore_list = []
    ore_type_list = []

    ############ ADD YOUR CODE HERE ############

    # Convert BGR -> HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # =========================================================
    # HSV RANGES
    # =========================================================

    # Blue -> Azurite
    azurite_lower = np.array([90, 120, 60])
    azurite_upper = np.array([120, 255, 255])

    # Green -> Malachite
    malachite_lower = np.array([60, 120, 80])
    malachite_upper = np.array([90, 255, 255])

    # Orange -> Vanadinite
    vanadinite_lower = np.array([0, 200, 100])
    vanadinite_upper = np.array([20, 255, 255])

    # CREATE MASKS
    # =========================================================

    azurite_mask = cv2.inRange(
        hsv,
        azurite_lower,
        azurite_upper
    )

    malachite_mask = cv2.inRange(
        hsv,
        malachite_lower,
        malachite_upper
    )

    vanadinite_mask = cv2.inRange(
        hsv,
        vanadinite_lower,
        vanadinite_upper
    )

    # MORPHOLOGICAL CLEANING
    # =========================================================

    kernel = np.ones((5, 5), np.uint8)

    azurite_mask = cv2.morphologyEx(
        azurite_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    azurite_mask = cv2.morphologyEx(
        azurite_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    malachite_mask = cv2.morphologyEx(
        malachite_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    malachite_mask = cv2.morphologyEx(
        malachite_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    vanadinite_mask = cv2.morphologyEx(
        vanadinite_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    vanadinite_mask = cv2.morphologyEx(
        vanadinite_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    ore_masks = [
        (azurite_mask, "azurite_ore"),
        (malachite_mask, "malachite_ore"),
        (vanadinite_mask, "vanadinite_ore")
    ]

    for mask, ore_name in ore_masks:

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            # Area of detected region
            area = cv2.contourArea(contour)

            print(f"{ore_name}: contour area = {area:.1f}"
            )

            # Ignore small noise
            if area < 150:
                continue

            # Calculate centroid
            M = cv2.moments(contour)

            if M["m00"] == 0:
                continue

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Store detection
            center_ore_list.append((cX, cY))
            ore_type_list.append(ore_name)

            # Draw contour
            cv2.drawContours(
                image,
                [contour],
                -1,
                (0, 0, 0),
                2
            )

            # Draw centroid
            cv2.circle(
                image,
                (cX, cY),
                6,
                (0, 0, 0),
                -1
            )

            # Draw ore label
            cv2.putText(
                image,
                ore_name,
                (cX - 50, cY - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2
            )

            # Draw pixel coordinates
            cv2.putText(
                image,
                f"({cX},{cY})",
                (cX - 50, cY + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1
            )

    return center_ore_list, ore_type_list

def assign_ore_ids(center_ore_list, ore_type_list, ore_tracks, max_match_distance):
    """
    Assign persistent IDs (1 or 2) to detected ores.

    IDs are maintained separately for each ore type.
    Matching is based on nearest previous pixel position.
    """

    ore_id_list = [None] * len(center_ore_list)

    # Process each ore type separately
    for ore_type in ore_types:

        # Get detections belonging to this ore type
        detections = []

        for i, detected_type in enumerate(ore_type_list):

            if detected_type == ore_type:
                detections.append(i)

        if len(detections) == 0:
            continue

        previous_positions = ore_tracks[ore_type]

        unmatched_detections = detections.copy()
        unused_ids = [1, 2]

        possible_matches = []

        # Calculate possible matches
        for detection_index in detections:

            cX, cY = center_ore_list[detection_index]

            for ore_id in [1, 2]:

                previous_center = previous_positions[ore_id]

                if previous_center is None:
                    continue

                px, py = previous_center

                distance = math.sqrt(
                    (cX - px) ** 2 +
                    (cY - py) ** 2
                )

                if distance <= max_match_distance:

                    possible_matches.append(
                        (distance, detection_index, ore_id)
                    )

        # Closest matches first
        possible_matches.sort(key=lambda x: x[0])

        # Assign unique existing IDs
        for distance, detection_index, ore_id in possible_matches:

            if detection_index not in unmatched_detections:
                continue

            if ore_id not in unused_ids:
                continue

            ore_id_list[detection_index] = ore_id

            unmatched_detections.remove(detection_index)
            unused_ids.remove(ore_id)

        # Assign remaining detections to unused IDs
        for detection_index in unmatched_detections:

            if len(unused_ids) == 0:
                break

            ore_id = unused_ids.pop(0)

            ore_id_list[detection_index] = ore_id

        # Update previous positions
        for detection_index in detections:

            ore_id = ore_id_list[detection_index]

            if ore_id is not None:

                ore_tracks[ore_type][ore_id] = \
                    center_ore_list[detection_index]

    return ore_id_list


##################### CLASS DEFINITION #######################

class ore_tf(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to detect the ores in the cell and
                    broadcast a transform for each one.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ore_tf
        '''

        super().__init__('ore_tf_publisher')                                            # registering node

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(Image, color_topic, self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, depth_topic, self.depthimagecb, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.caminfocb, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        image_processing_rate = 0.2                                                     # rate of time to process image (seconds)
        self.bridge = CvBridge()                                                        # initialise CvBridge object for image conversion
        self.tf_buffer = tf2_ros.buffer.Buffer()                                        # buffer time used for listening transforms
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)                                    # object as transform broadcaster to send transform wrt some frame_id
        self.timer = self.create_timer(image_processing_rate, self.process_image)       # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)

        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None   
        self.depth_frame = None                                                      # depth image variable (from depthimagecb())
        self.cam_info = None     
        
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
                                                               # camera intrinsics variable (from caminfocb())

        ############ ADD YOUR CODE HERE ############

        # Store the previous pixel position of each ore ID.
        # Each ore type has exactly two ores.
        self.ore_tracks = {
            'azurite_ore': {
                1: None,
                2: None
            },
            'malachite_ore': {
                1: None,
                2: None
            },
            'vanadinite_ore': {
                1: None,
                2: None
            }
        }

        # Maximum pixel distance allowed when matching
        # an ore in the current frame to its previous position.
        self.max_match_distance = 80.0

        ############################################


    def depthimagecb(self, data):
        '''
        Description:    Callback function for the aligned depth camera topic.
                        Use this function to receive the depth image and convert it to a CV2 image.

        Args:
            data (Image):    Input depth image frame received from the aligned depth camera topic

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:

            depth = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            self.depth_frame = data.header.frame_id

            if depth.dtype == np.uint16:

                depth = depth.astype(np.float32) / 1000.0

            elif depth.dtype == np.float32:

                depth = depth.astype(np.float32)

            else:

                depth = depth.astype(np.float32)

            self.depth_image = depth

        except CvBridgeError as e:

            self.get_logger().error(f"Error converting depth image: {e}")
        ############################################


    def colorimagecb(self, data):
        '''
        Description:    Callback function for the colour camera raw topic.
                        Use this function to receive the raw image and convert it to a CV2 image.

        Args:
            data (Image):    Input coloured raw image frame received from the image_raw camera topic

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:

            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')        # convert ROS Image message to CV2 image and store it

        except CvBridgeError as e:

            self.get_logger().error(f"error converting your image: {e}")

        ############################################


    def caminfocb(self, data):
        '''
        Description:    Callback function for the camera info topic.
                        Use this function to receive the camera's intrinsic parameters.

        Args:
            data (CameraInfo):    Camera calibration published by the camera

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        self.cam_info = data

        self.fx = float(data.k[0])
        self.fy = float(data.k[4])
        self.cx = float(data.k[2])
        self.cy = float(data.k[5])
        ############################################


    def process_image(self):
        '''
        Description:    Timer function used to detect the ores and publish a transform for
                        each one on its estimated position.

        Args:
        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        # Wait for colour image
        if self.cv_image is None:
            return

        # Wait for depth image
        if self.depth_image is None:
            return

        # Wait for camera information
        if self.cam_info is None:
            return

        if self.fx == 0.0 or self.fy == 0.0:
            return

        # Get the ore centers and their types
        center_ore_list, ore_type_list = detect_ores(self.cv_image)

        ore_id_list = assign_ore_ids(
            center_ore_list,
            ore_type_list,
            self.ore_tracks,
            self.max_match_distance
        )


        # Process every detected ore
        for i in range(len(center_ore_list)):

            cX, cY = center_ore_list[i]
            ore_type = ore_type_list[i]
            ore_id = ore_id_list[i]

            # -------------------------------------------------
            # Read depth around the ore center
            # -------------------------------------------------

            height, width = self.depth_image.shape

            x1 = max(0, cX - 4)
            x2 = min(width, cX + 5)

            y1 = max(0, cY - 4)
            y2 = min(height, cY + 5)

            patch = self.depth_image[y1:y2, x1:x2]

            # Keep only valid depth values
            valid_depth = patch[
                np.isfinite(patch) & (patch > 0.0)
            ]

            # Skip if no valid depth exists
            if valid_depth.size == 0:
                continue

            # Median depth in metres
            z = float(np.median(valid_depth))

            # -------------------------------------------------
            # Deproject pixel + depth into 3D camera frame
            # -------------------------------------------------

            u = float(cX)
            v = float(cY)

            x = (u - self.cx) * z / self.fx
            y = (v - self.cy) * z / self.fy

            point_in_camera = PointStamped()

            point_in_camera.header.stamp = self.get_clock().now().to_msg()
            point_in_camera.header.frame_id = self.depth_frame

            point_in_camera.point.x = float(x)
            point_in_camera.point.y = float(y)
            point_in_camera.point.z = float(z)

            try:
                tf = self.tf_buffer.lookup_transform(
                    base_frame,
                    self.depth_frame,
                    rclpy.time.Time()
                )

                point_in_base = do_transform_point(
                    point_in_camera,
                    tf
                )

            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException
            ):
                continue

            # Print the 3D camera coordinates
            self.get_logger().info(
                f'{ore_type}_{ore_id}: '
                f'base_xyz=('
                f'{point_in_base.point.x:.3f}, '
                f'{point_in_base.point.y:.3f}, '
                f'{point_in_base.point.z:.3f}) m'
            )

            # Broadcast ore TF
            t = TransformStamped()

            t.header.stamp = point_in_base.header.stamp
            t.header.frame_id = base_frame
            t.child_frame_id = f'{ore_type}_{ore_id}'

            t.transform.translation.x = float(point_in_base.point.x)
            t.transform.translation.y = float(point_in_base.point.y)
            t.transform.translation.z = float(point_in_base.point.z)

            # Identity orientation
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            self.br.sendTransform(t)
    ############################################


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the ore_tf
                    class to perform its task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('ore_tf_process')                      # creating ROS node

    node.get_logger().info('Node created: Ore tf process')          # logging information

    ore_tf_class = ore_tf()                                         # creating a new object for class 'ore_tf'

    rclpy.spin(ore_tf_class)                                        # spining on the object to make it alive in ROS 2 DDS

    ore_tf_class.destroy_node()                                     # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':

    main()