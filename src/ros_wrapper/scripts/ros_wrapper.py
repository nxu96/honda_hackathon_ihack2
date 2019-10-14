#!/usr/bin/env python2
import rospy
import message_filters
import cv2
import base64
import sys
from math import atan2
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from novatel_gps_msgs.msg import NovatelUtmPosition
from cv_bridge import CvBridge, CvBridgeError
from functools import partial
import zmq
import json
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

def callbackFunc(timestamp, back_car_image, back_car_pose, back_car_heading, side_car_image, side_car_pose, side_car_heading):
  print("{} {} {} {} {}".format(timestamp, back_car_pose, back_car_heading, side_car_pose, side_car_heading))

class RosWrapper:
    def __init__(self, callback):
        self.callback = callback

        self.bridge = CvBridge()
        rospy.init_node('ros_wrapper', anonymous=True)

        run_number = rospy.get_param("/run_number")
        
        # back_car_image_sub = message_filters.Subscriber("/back_car/front_usb_cam/image_raw", Image)
        # back_car_odom_sub = message_filters.Subscriber("/back_car/bestutm", NovatelUtmPosition)
        # side_car_image_sub = message_filters.Subscriber("/side_car/front_usb_cam/image_raw", Image)
        # side_car_odom_sub = message_filters.Subscriber("/side_car/odom", Odometry)

        back_car_image_sub = rospy.Subscriber("/back_car/front_usb_cam/image_raw", Image, self.backCarCallbackWrapper)
        back_car_odom_sub = rospy.Subscriber("/back_car/bestutm", NovatelUtmPosition, self.backCarUTMCallback)
        side_car_image_sub = rospy.Subscriber("/side_car/front_usb_cam/image_raw", Image, self.sideCarCallbackWrapper)
        side_car_odom_sub = rospy.Subscriber("/side_car/odom", Odometry, self.sideCarUTMCallback)

        # ts1 = message_filters.ApproximateTimeSynchronizer([back_car_image_sub, back_car_odom_sub], 1, 10)
        # tmp_callback1 = partial(self.backCarCallbackWrapper)
        # ts1.registerCallback(tmp_callback1)

        # ts2 = message_filters.ApproximateTimeSynchronizer([side_car_image_sub, side_car_odom_sub], 1, 10)
        # tmp_callback2 = partial(self.sideCarCallbackWrapper)
        # ts2.registerCallback(tmp_callback2)
        
        # TODO Align timestamps to when messages are first coming in from both bags
        if run_number == 1:
            self.back_car_prev_pose = [ 308839.1992 - (308838.9261 - 308839.1992), 4662374.7451 - (4662374.7376 - 4662374.7451) ]
            self.side_car_prev_pose = [ 308829.66194 - (308828.28348 - 308829.66194), 4662370.78086 - (4662370.81815 - 4662370.78086) ]
        elif run_number == 2:
            self.back_car_prev_pose = [ 308372.0384 - (308373.0953 - 308372.0384), 4662374.4116 - (4662374.9749 - 4662374.4116) ]
            self.side_car_prev_pose = [ 308371.905587 - (308372.46701 - 308371.905587), 4662384.29252 - (4662384.6477 - 4662384.29252) ]
        elif run_number == 3:
            self.back_car_prev_pose = [ 308519.8711 - (308521.0738 - 308519.8711), 4662453.5285 - (4662454.1876 - 4662453.5285) ]
            self.side_car_prev_pose = [ 308371.905587 - (308372.46701 - 308371.905587), 4662384.29252 - (4662384.6477 - 4662384.29252) ]
            # Time stamp alignment would make it:
            # self.side_carprev_pose = [ 308544.417269 - (308550.434833 - 308544.417269), 4662473.69861 - (4662476.8692 - 4662473.69861) ]

        self.back_car_prev_heading = 0
        self.side_car_prev_heading = 0

        self.back_car_utm_loc = None
        self.side_car_utm_loc = None

        # ZMQ stuffs
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://127.0.0.1:5555")

    def mogrify(self, topic, image, msg):
        """ json encode the message and prepend the topic """
        return topic + '?' + image + '?' + json.dumps(msg)

    def backCarUTMCallback(self, back_car_utm):
        self.back_car_utm_loc = back_car_utm

    def backCarCallbackWrapper(self, back_car_image):
        try:
            back_car_cv_image = self.bridge.imgmsg_to_cv2(back_car_image, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        if self.back_car_utm_loc is None:
            return

        back_car_pose = [ self.back_car_utm_loc.easting, self.back_car_utm_loc.northing ]

        now = rospy.get_rostime()
        timestamp = [ now.secs, now.nsecs ]

        back_car_heading = atan2(back_car_pose[0] - self.back_car_prev_pose[0], back_car_pose[1] - self.back_car_prev_pose[1])
        if back_car_pose[0] - self.back_car_prev_pose[0] == 0 and back_car_pose[1] - self.back_car_prev_pose[1] == 0:
            back_car_heading = self.back_car_prev_heading
        self.back_car_prev_pose = back_car_pose
        self.back_car_prev_heading = back_car_heading

        retval, image_buffer = cv2.imencode('.jpg', back_car_cv_image)
        serialized_image = base64.b64encode(image_buffer)

        # print("Back Car: {} {}".format(back_car_pose, back_car_heading))
        
        #self.socket.send(self.mogrify('b', {"timestamp": timestamp, "back_car_image": serialized_image, "back_car_pose": back_car_pose, "back_car_heading": back_car_heading}))

        serialized_image = pickle.dumps(back_car_cv_image, protocol=2)
        self.socket.send_multipart(['b', serialized_image, json.dumps({"timestamp": timestamp, "back_car_pose": back_car_pose, "back_car_heading": back_car_heading})])
        
        #self.callback(timestamp, back_car_cv_image, back_car_pose, back_car_heading, side_car_cv_image, side_car_pose, side_car_heading)

    def sideCarUTMCallback(self, side_car_utm):
        self.side_car_utm_loc = side_car_utm

    def sideCarCallbackWrapper(self, side_car_image):
        try:
            side_car_cv_image = self.bridge.imgmsg_to_cv2(side_car_image, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        if self.side_car_utm_loc is None:
            return

        side_car_pose = [ self.side_car_utm_loc.pose.pose.position.x, self.side_car_utm_loc.pose.pose.position.y ]

        now = rospy.get_rostime()
        timestamp = [ now.secs, now.nsecs ]

        side_car_heading = atan2(side_car_pose[0] - self.side_car_prev_pose[0], side_car_pose[1] - self.side_car_prev_pose[1])
        if side_car_pose[0] - self.side_car_prev_pose[0] == 0 and side_car_pose[1] - self.side_car_prev_pose[1] == 0:
            side_car_heading = self.side_car_prev_heading
        self.side_car_prev_pose = side_car_pose
        self.side_car_prev_heading = side_car_heading

        # print("Side Car: {} {}".format(side_car_pose, side_car_heading))

        #self.socket.send(self.mogrify('s', serialized_image, {"timestamp": timestamp, "side_car_pose": side_car_pose, "side_car_heading": side_car_heading}))
        
        serialized_image = pickle.dumps(side_car_cv_image, protocol=2)
        self.socket.send_multipart(['s', serialized_image, json.dumps({"timestamp": timestamp, "side_car_pose": side_car_pose, "side_car_heading": side_car_heading})])

        #self.callback(timestamp, back_car_cv_image, back_car_pose, back_car_heading, side_car_cv_image, side_car_pose, side_car_heading)

if __name__ == "__main__":
    RosWrapper(callbackFunc)
    rospy.spin()
