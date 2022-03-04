#!/usr/bin/env python
"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""

import os
import time

import pycuda.autoinit  # This is needed for initializing CUDA driver
import pycuda.driver as cuda

# Python libs
import sys, time

# numpy and scipy
import numpy as np
from scipy.ndimage import filters

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy
from rospkg import RosPack

# Ros Messages
from sensor_msgs.msg import CompressedImage
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import get_input_shape, TrtYOLO

WINDOW_NAME = 'TRT_Object_Detection'
VERBOSE=False
package = RosPack()
package_path = package.get_path('yolo_trt_ros')

class yolo_ros(object):

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.image_pub = rospy.Publisher("/output/image/compressed", CompressedImage, queue_size = 1)
        # self.bridge = CvBridge() 

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/usb_cam/image_raw/compressed",
            CompressedImage, self.callback,  queue_size = 1) # subscriber Usbcamera image
        model = 'yolov4-tiny-416' # load yolov4-tiny-416.trt
        category_num = 80
        letter_box = False
        
        if category_num <= 0:
            raise SystemExit('ERROR: bad category_num (%d)!' % category_num)
        if not os.path.isfile('yolo/%s.trt' % model):
            raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % model)

        self.cls_dict = get_cls_dict(category_num)
        self.vis = BBoxVisualization(self.cls_dict)
        h, w = get_input_shape(model)

        self.cuda_ctx = cuda.Device(0).make_context()
        self.trt_yolo = TrtYOLO(model, (h, w), category_num, letter_box)

        if VERBOSE :
            print("subscribed to /usb_cam/image_raw/compressed")

    def __del__(self):
        """ Destructor """
        
        self.cuda_ctx.pop()
        del self.trt_yolo
        del self.cuda_ctx

    def clean_up(self):
        """ Backup destructor: Release cuda memory """

        if self.trt_yolo is not None:
            self.cuda_ctx.pop()
            del self.trt_yolo
            del self.cuda_ctx


    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE :
            print('received image of type: "%s"' % ros_data.format)

        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        
        # cv_img = self.bridge.imgmsg_to_cv2(ros_img, desired_encoding="bgr8")
        if image_cv is not None:
            self.cuda_ctx.push()
            boxes, confs, clss = self.trt_yolo.detect(image_cv, conf_th=0.3)
            self.cuda_ctx.pop()
            img = self.vis.draw_bboxes(image_cv, boxes, confs, clss)
            cv2.imshow(WINDOW_NAME, img)
            cv2.waitKey(1)

        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_cv)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)
        
        #self.subscriber.unregister()

def main():
    
    '''Initializes and cleanup ros node'''
    yolo = yolo_ros()
    rospy.init_node('yolo_ros', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.on_shutdown(yolo.clean_up())
        print("Shutting down ROS Yolo detector module")
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
