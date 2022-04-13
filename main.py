## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
from scipy.spatial import KDTree
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb
import pandas as pd


#css3_db = CSS3_HEX_TO_NAMES
names = ["yellow", "orange", "red", "green", "blue", "purple", "background", "background2"]
rgb_values = [[254,231,31],[254,142,31], [241,33,17],[25,171,37],[16,119,161],[184,55,185],[255,255,255],[0,0,0]]
#for color_hex, color_name in css3_db.items():
    #names.append(color_name)
    #rgb_values.append(hex_to_rgb(color_hex))


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

def main(filepath):
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

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

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
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            color_image_2d = np.reshape(color_image,(307200,3))
            cv2.imwrite('original_image_1.jpeg', color_image)
            closest_colors = []
            for i in range(len(color_image_2d)):
                closest = convert_rgb_to_closest(color_image_2d[i])
                closest_colors.append(closest)
            closest_colors = np.array(closest_colors)
            closest_colors = np.reshape(closest_colors,(480,640,3))
            print(closest_colors)
            print(closest_colors.shape)
            cv2.imwrite('closest_image_output_1.jpeg', closest_colors)



            #np.save("color-data",color_image)
            #color_image_2d = np.reshape(color_image,(307200,3))
            #ci = pd.DataFrame(color_image_2d)
            #ci.to_csv("color_image.csv")

            #print("color: ", color_image)
            # print("depth: ", depth_image)
            

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            #print(color_image.shape)
            # dimensions: 480, 640, 3
            #print("0,0: ", convert_rgb_to_names(color_image[0][0]))
            #print("240,320: ", convert_rgb_to_names(color_image[240][320]))
            #print("100,150: ", convert_rgb_to_names(color_image[100][150]))
            #print("50,90: ", convert_rgb_to_names(color_image[50][90]))
            #print("300,580: ", convert_rgb_to_names(color_image[300][580]))
        


            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            Key = cv2.waitKey(1)
            if Key == 27:
                break

            banana = False

    finally:

        # Stop streaming
        pipeline.stop()

if __name__ == '__main__':
    main("")
