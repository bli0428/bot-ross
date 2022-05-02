import colorsys
from sre_constants import GROUPREF_EXISTS
import cv2
import numpy as np
import imutils
from wlkata_mirobot import WlkataMirobot
from wlkata_mirobot import WlkataMirobotTool
import pyrealsense2 as rs
import pandas as pd
from preprocessing import downsample, pull_from_file


# 1. Establish range of H values for colors: red, orange, yellow, blue, green purple, light background (white), dark background (black)
# Create a default rgb tuple for each of those colors for visualization purposes.
names = ["red","orange","yellow","green","blue","purple","background_white", "background_black"]
end_locations = [[[229.9, 128.4, 19.8], [229.5, 103.1, 17.9], [229.4, 72.9, 16.1], [229.2, 47.7, 15.1]], 
[[203.4, 132, 19.2],[203.3, 106.8, 17.3], [203, 76.4, 15.5], [202.7, 51.2, 14]], [[176.5, 131.9, 18], 
[176.2, 106.6, 15.8],[173, 81.1, 16.7], [173, 50.8, 15.5]], [[144.9, 135, 14.1], [144.5, 109.9, 17.6], 
[144.2, 79.7, 16], [143.8, 52.8, 16.7]]]
arm = WlkataMirobot(portname='/dev/cu.usbserial-1410')
# H_ranges = 
# while color ranges can be determined by the hue value, 
# might need to check black/white another way because white seems to be when saturation is like <.25
# and black seems to be when value is below <.35 

def get_HSVcolor(h,s,v):
    #pixel = img[cx][cy]
    #hsv = colorsys.rgb_to_hsv(pixel[0]/255,pixel[1]/255,pixel[2]/255)
    #hsv = colorsys.rgb_to_hsv(r/255,g/255,b/255)
    #hsv = colorsys.rgb_to_hsv(r,g,b)
    hsv = [h,s,v]
    print("Hue: ", hsv[0], " converted: ",    hsv[0]*360)
    print("Saturation: ", hsv[1])
    print("Value: ", hsv[2])
    if hsv[1] < .13:
        return default_rgb_values[6], names[6]
    elif hsv[2] < .13:
        return default_rgb_values[7], names[7]
    else:
        if hsv[0] <= 18/360 or hsv[0] >= 342/360: #red
            return default_rgb_values[0], names [0]
        elif 18/360 < hsv[0] <= 38/360 : #orange
            return default_rgb_values[1], names [1]
        elif 38/360 < hsv[0] <= 66/360 : #yellow
            return default_rgb_values[2], names [2]
        elif 67/360 < hsv[0] <= 169/360 : #green
            return default_rgb_values[3], names [3]
        elif 169/360 < hsv[0] <= 257/360 : #blue
            return default_rgb_values[4], names [4]
        else:
            return default_rgb_values[5], names [5]
        

def block_picking(img, color, location): 
    # 2. import one of the output camera files ("original_image...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("b/w", gray)
    cv2.waitKey(0)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #for use with non-black background
    #thresh = cv2.threshold(blurred, 1.86*gray[0][0]-237.87, 255, cv2.THRESH_BINARY_INV)[1]
    #for use with black background
    thresh = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)[1]
    #second value should depend on overall brightness

    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    image = img

    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / (M["m00"]+ 1e-5))
        cY = int(M["m01"] / (M["m00"]+ 1e-5))
        # draw the contour and center of the shape on the image
        # rgb_val, name = get_HSVcolor(cX,cY)
        

        #GETTING AVERAGE OF 400 pixels around center
        r = 0
        g = 0
        b = 0
        h = 0
        s = 0
        v = 0
        for i in range(-10,10):
            for j in range(-10,10):
                newX = min(len(img[0])-1,max(0,cX+i))
                newY = min(len(img)-1,max(0,cY+i))
                pixel = img[newY][newX]
                r += pixel[0]
                g += pixel[1]
                b += pixel[2]
                pixel_hsv = img_hsv[newY][newX]
                h += pixel_hsv[0]
                s += pixel_hsv[1]
                v += pixel_hsv[2]
        r /= 400
        g /= 400
        b /= 400
        h /= 400*180
        s /= 400*360
        v /= 400*360

        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 7, (int(r), int(g), int(b)), -1)
        rgb_val, name = get_HSVcolor(h,s,v)
        if name == color:
            real_y = -.4105*cX - 17.67136 #-.62*cX-26.44 #-.682 - NEED TO RECALIBRATE THIS
            real_x = -.405*cY + 282.26 #-.52*cY+332.238 #.49
            arm.set_tool_pose(real_x,real_y,10)
            arm.pump_suction()
            arm.set_tool_pose(real_x,real_y, 45)
            arm.set_tool_pose(location[0], location[1], location[2]) # NOTE CAN CHANGE LAST VALUE TO 10 if we don't want to use the different depth values 
            arm.pump_blowing()
            arm.set_tool_pose(location[0], location[1], 45)

        cv2.putText(image, name, (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print("Y: ", cY, "X:", cX)
        # show the image
        cv2.imshow("Image", image)
        cv2.waitKey(0)


def main(file):
    arm.home()
    arm.set_tool_type(WlkataMirobotTool.SUCTION_CUP)
    
    img = pull_from_file(file) 
    output = downsample(img,4,4)

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

    try:
        for i in range(4):
            for j in range(4):

                [r,g,b] = output[i][j]
                h,s,v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                rgb, name = get_HSVcolor(h,s,v)

                color = name
                end_location = end_locations[i][j]

                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                #color_image_2d = np.reshape(color_image,(307200,3))

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape

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

                block_picking(color_image, color, end_location)

    finally:

        # Stop streaming
        pipeline.stop()

if __name__ == '__main__':
    main("images/pink_flower.jpeg")