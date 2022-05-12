## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
from scipy.spatial import KDTree
#from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb
import pandas as pd
from wlkata_mirobot import WlkataMirobot



#css3_db = CSS3_HEX_TO_NAMES
names = ["yellow", "orange", "red", "green", "blue", "purple", "background", "background2"]
rgb_values = [[254,231,31],[254,142,31], [241,33,17],[25,171,37],[16,119,161],[184,55,185],[255,255,255],[0,0,0]]


def convert_rgb_to_names(rgb_tuple):
    # a dictionary of all the hex and their respective names in css3
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return f'closest match: {names[index]}'

def convert_rgb_to_closest(rgb_tuple):
    # a dictionary of all the hex and their respective names in css3
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return rgb_values[index]


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

banana = True

try:
    while banana:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        #depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image_2d = np.reshape(color_image,(307200,3))
        cv2.imwrite('calibration_may12.jpeg', color_image)
        
        #depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        images = color_image

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        Key = cv2.waitKey(1)
        if Key == 27:
            break

        #banana = False

finally:

    # Stop streaming
    pipeline.stop()
