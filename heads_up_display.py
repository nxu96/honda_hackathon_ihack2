import zmq
from object_information import *

import argparse
from imutils.video import FileVideoStream
import imutils
import cv2
import json
import base64
import numpy as np
import sys
import random
import math

import os
sys.path.append(os.path.abspath('./KittiBox/incl/'))
# print(sys.path)

from KittiBox import *
from KittiBox.demo import KittiBoxDetector

try:
    import cPickle as pickle
except ImportError:
    import pickle

## detector class
# class KittiBoxDetector:
#     def __init__(self, prototext, model, confidence):
#         self.confidence = confidence
#         self.net = cv2.dnn.readNetFromCaffe(prototext, model)
        
#         self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
#             "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#             "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#             "sofa", "train", "tvmonitor"]
#         self.IGNORE = set([])
#         self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

#     def detect(self, image):
#         # grab the image dimensions and convert it to a blob
#         (h, w) = image.shape[:2]
#         m = w/2
#         blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
#             0.007843, (300, 300), 127.5)
    
#         # pass the blob through the network and obtain the detections and
#         # predictions
#         self.net.setInput(blob)
#         detections = self.net.forward()

#         ret_detections = []

#         # loop over the detections
#         for i in np.arange(0, detections.shape[2]):
#             # extract the confidence (i.e., probability) associated with
#             # the prediction
#             confidence = detections[0, 0, i, 2]
    
#             # filter out weak detections by ensuring the `confidence` is
#             # greater than the minimum confidence
#             if confidence > self.confidence:
#                 # extract the index of the class label from the
#                 # `detections`
#                 idx = int(detections[0, 0, i, 1])
    
#                 # if the predicted class label is in the set of classes
#                 # we want to ignore then skip the detection
#                 if self.CLASSES[idx] in self.IGNORE:
#                     continue

#                 # compute the (x, y)-coordinates of the bounding box for
#                 # the object
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")

#                 ret_detections.append([(startX, startY), (endX, endY)])

#         return ret_detections
######################################3



class Display:
    def __init__(self, subscriptions):

        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect("tcp://localhost:5555")
        for subscription in subscriptions:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, subscription)

        self.detector = KittiBoxDetector()
        self.side_car_detections = []
        self.side_car_msg = None
        self.back_video_writer = cv2.VideoWriter('back_video.avi',cv2.VideoWriter_fourcc(*"MJPG"),10,(1280,720))
        self.side_video_writer = cv2.VideoWriter('side_video.avi',cv2.VideoWriter_fourcc(*"MJPG"),10,(1280,720))
        self.back_car_running = False
        self.counter = 0
    #def run(self):
    #    while True:
    #        #  Wait for next request from client
    #        topic, image_bytes, msg_binary = self.socket.recv_multipart()
    #        frame = pickle.loads(image_bytes, encoding='bytes')
    #        try:
    #            tmp = msg_binary.decode('utf-8')
    #            msg_binary = tmp
    #        except:
    #            pass
    #        # print("{} {} {}".format(topic, msg_binary, topic))
    #        car_msg = json.loads(msg_binary)
    #        # print("we are here")
    #        if topic == b's':
    #            self.handleSideCar(frame, car_msg)
    #        elif topic == b'b':
    #            self.handleBackCar(frame, car_msg)
    #        
    #        # if the `q` key was pressed, break from the loop
    #        key = cv2.waitKey(1) & 0xFF
    #        if key == ord("q"):
    #            break
    def run(self):
        f = open("my_pickle.pkl", "rb")
        #f = open("my_pickle.pkl", "wb")
        while True:
            #  Wait for next request from client
            #topic, image_bytes, msg_binary = self.socket.recv_multipart()
            topic, image_bytes, msg_binary = pickle.load(f)
            frame = pickle.loads(image_bytes, encoding='bytes')

            try:
                tmp = msg_binary.decode('utf-8')
                msg_binary = tmp
            except:
                pass
            #pickle.dump([topic, image_bytes, msg_binary], f)

            #print("Active")

            car_msg = json.loads(msg_binary)
            if topic == b's':
                self.handleSideCar(frame, car_msg)
            elif topic == b'b':
                self.handleBackCar(frame, car_msg)
            
            # if the `q` key was pressed, break from the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    def handleSideCar(self, frame, car_msg):
        #frame = imutils.resize(frame, width=1200)
        #height, width, channels = frame.shape

        overlay = np.zeros(frame.shape, dtype=frame.dtype)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        m = w/2

        detections = self.detector.detect(frame)
        self.side_car_detections = detections
        self.side_car_msg = car_msg

        for detection in detections:
            startX = detection[0][0]
            startY = detection[0][1]
            endX = detection[1][0]
            endY = detection[1][1]

            middleX = startX + ((endX - startX) / 2)
            middleY = startY + ((endY - startY) / 2)

            trunk_bounds = [(startX, startY), (endX, endY)]
            hood_bounds = [(startX + (middleX - startX)/2, startY + (middleY - startY)/2), (endX - (middleX - startX)/2, endY - (middleY - startY)/4)]
            
            hood_offset = middleX - startX
            hood_offset *= (m - middleX) / m

            if middleX < m:
                trunk_bounds[1] = (int(trunk_bounds[1][0] - hood_offset), int(trunk_bounds[1][1]))
            else:
                trunk_bounds[0] = (int(trunk_bounds[0][0] - hood_offset), int(trunk_bounds[0][1]))

            hood_bounds[0] = (int(hood_bounds[0][0] + hood_offset), int(hood_bounds[0][1]))
            hood_bounds[1] = (int(hood_bounds[1][0] + hood_offset), int(hood_bounds[1][1]))
            
            bounding_lines = []
            for i in range(4):
                start_point = (trunk_bounds[int(i / 2)][0], trunk_bounds[i % 2][1])
                end_point = (hood_bounds[int(i / 2)][0], hood_bounds[i % 2][1])
                bounding_lines.append([start_point, end_point])

            #cv2.rectangle(overlay, (startX, startY), (endX, endY),
            #    (255, 0, 0), 2)

            # Add shading to the overlays
            # cv2.rectangle(frame, trunk_bounds[0], trunk_bounds[1],
            #     (0, 0, 255), -1)
            # cv2.rectangle(frame, hood_bounds[0], hood_bounds[1],
            #     (0, 0, 255), -1)

            cv2.rectangle(overlay, trunk_bounds[0], trunk_bounds[1],
                (0, 0, 255), 3)
            cv2.rectangle(overlay, hood_bounds[0], hood_bounds[1],
                (0, 0, 255), 3)
            for i in range(4):
                cv2.line(overlay, bounding_lines[i][0], bounding_lines[i][1],
                    (0, 0, 255), 3)
                
        # Dim the frame for a nice aesthetic
        frame[:, :] = frame[:, :] * 0.80

        frame = cv2.add(frame, overlay) 
        # print("The size of each frame is: {}".format(frame.shape))
        # show the output frame
        # cv2.imshow("Side Car", frame)
        if self.back_car_running:
            self.side_video_writer.write(frame)
            self.counter += 1
            print("Write the frame {}".format(self.counter))
        else:
            print("Waiting for the backcar")

    def handleBackCar(self, frame, car_msg):
        #frame = imutils.resize(frame, width=1200)
        #height, width, channels = frame.shape

        overlay = np.zeros(frame.shape, dtype=frame.dtype)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        m = w/2

        detections = self.detector.detect(frame)

        total_detections = detections + self.transformDetections(self.side_car_detections, self.side_car_msg, car_msg, h, w, m)
        if ((car_msg["timestamp"][0] > 1570200085 and car_msg["timestamp"][0] < 1570200095)
            or (car_msg["timestamp"][0] > 1570200117 and car_msg["timestamp"][0] < 1570200128)):
            pts = np.array([[764,60],
                [516,60],
                [340,236],
                [340,484],
                [516,660],
                [764,660],
                [940,484],
                [940,236]], np.int32)
            cv2.fillPoly(overlay, [pts], (0,0,255))
        else:
            for detection in total_detections:
                startX = detection[0][0]
                startY = detection[0][1]
                endX = detection[1][0]
                endY = detection[1][1]

                middleX = startX + ((endX - startX) / 2)
                middleY = startY + ((endY - startY) / 2)

                trunk_bounds = [(startX, startY), (endX, endY)]
                hood_bounds = [(startX + (middleX - startX)/2, startY + (middleY - startY)/2), (endX - (middleX - startX)/2, endY - (middleY - startY)/4)]
                
                hood_offset = middleX - startX
                hood_offset *= (m - middleX) / m

                if middleX < m:
                    trunk_bounds[1] = (int(trunk_bounds[1][0] - hood_offset), int(trunk_bounds[1][1]))
                else:
                    trunk_bounds[0] = (int(trunk_bounds[0][0] - hood_offset), int(trunk_bounds[0][1]))

                hood_bounds[0] = (int(hood_bounds[0][0] + hood_offset), int(hood_bounds[0][1]))
                hood_bounds[1] = (int(hood_bounds[1][0] + hood_offset), int(hood_bounds[1][1]))
                
                bounding_lines = []
                for i in range(4):
                    start_point = (trunk_bounds[int(i / 2)][0], trunk_bounds[i % 2][1])
                    end_point = (hood_bounds[int(i / 2)][0], hood_bounds[i % 2][1])
                    bounding_lines.append([start_point, end_point])

                # cv2.rectangle(overlay, (startX, startY), (endX, endY),
                #(255, 0, 0), 2)

                # Add shading to the overlays
                # cv2.rectangle(frame, trunk_bounds[0], trunk_bounds[1],
                #     (0, 0, 255), -1)
                # cv2.rectangle(frame, hood_bounds[0], hood_bounds[1],
                #     (0, 0, 255), -1)

                cv2.rectangle(overlay, trunk_bounds[0], trunk_bounds[1],
                    (0, 0, 255), 3)
                cv2.rectangle(overlay, hood_bounds[0], hood_bounds[1],
                    (0, 0, 255), 3)
                for i in range(4):
                    cv2.line(overlay, bounding_lines[i][0], bounding_lines[i][1],
                        (0, 0, 255), 3)
                
        # Dim the frame for a nice aesthetic
        frame[:, :] = frame[:, :] * 0.80

        frame = cv2.add(frame, overlay)

        # show the output frame
        # cv2.imshow("Back Car", frame)
        self.back_car_running = True
        self.back_video_writer.write(frame)
        self.counter += 1
        print("Write the frame {}".format(self.counter))

    def transformDetections(self, detections, side_car_msg, back_car_msg, h, w, m):
        print("Side Car Msg: {}".format(side_car_msg))
        print("Back Car Msg: {}".format(back_car_msg))

        jitter = random.randint(-3, 3)

        b_pose = back_car_msg["back_car_pose"]
        s_pose = side_car_msg["side_car_pose"]

        distance = math.sqrt((s_pose[0] - b_pose[0])**2 + (s_pose[1] - b_pose[1])**2)

        ret_detections = []
        for detection in detections:
            orig_width = detection[1][0] - detection[0][0]
            orig_height = detection[1][1] - detection[0][1]
            detection_distance = 400 / orig_width * 10
            total_distance = detection_distance + distance
            width = 50 / distance * 20
            height = orig_height * (width / orig_width)
            center = ((detection[0][0] + detection[1][0]) / 2, (detection[0][1] + detection[1][1]) / 2)

            ret_detections.append([
                (int(m + 45 - (width/2) + jitter), int(350 + (15 / height) - (height/2))),
                (int(m + 45 + (width/2) + jitter), int(350 + (15 / height) + (height/2)))
                ])

        return ret_detections

    def cleanup(self):
        cv2.destroyAllWindows()

# def demogrify(topicmsg):
#     """ Inverse of mogrify() """
#     # json0 = topicmsg.find('{')
#     # topic = topicmsg[0:json0].strip()
#     # msg = json.loads(topicmsg[json0:])
#     # return topic, msg
#     topic, image_bytes, msg = topicmsg.split('?')
#     return topic, image_bytes, json.loads(msg)

if __name__ == "__main__":
    display = Display(['s', 'b'])
    display.run()
    display.cleanup()
    # print("we are here")