import colorsys
from logging.config import valid_ident
from sre_constants import GROUPREF_EXISTS
import cv2
import numpy as np
import imutils
from wlkata_mirobot import WlkataMirobot
from wlkata_mirobot import WlkataMirobotTool
import pyrealsense2 as rs
import pandas as pd
from preprocessing import downsample, pull_from_file
import time


# 1. Establish range of H values for colors: red, orange, yellow, blue, green purple, light background (white), dark background (black)
# Create a default rgb tuple for each of those colors for visualization purposes.
names = ["red","orange","yellow","green","blue","purple","background_white", "background_black"]
default_rgb_values = [[254,231,31],[254,142,31], [241,33,17],[25,171,37],[16,119,161],[184,55,185],[255,255,255],[0,0,0]]
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
    #print("Saturation: ", hsv[1])
    #print("Value: ", hsv[2])
    if hsv[1] < .13:
        return default_rgb_values[6], names[6]
    elif hsv[2] < .13:
        return default_rgb_values[7], names[7]
    else:
        if 0 <= hsv[0] <= 18/360 or 1 >= hsv[0] >= 328/360: #red - was above 331
            return default_rgb_values[0], names [0]
        elif 18/360 < hsv[0] <= 42/360 : #orange
            return default_rgb_values[1], names [1]
        elif 39/360 < hsv[0] <= 66/360 : #yellow
            return default_rgb_values[2], names [2]
        elif 67/360 < hsv[0] <= 169/360 : #green
            return default_rgb_values[3], names [3]
        elif 170/360 < hsv[0] <= 257/360 : #blue
            return default_rgb_values[4], names [4]
        else:
            return default_rgb_values[5], names [5]
        

def block_picking(img, color, location): 
    # 2. import one of the output camera files ("original_image...")
    print("in block picking")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("b/w", gray)
    #cv2.waitKey(0)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #for use with non-black background
    #thresh = cv2.threshold(blurred, 1.86*gray[0][0]-237.87, 255, cv2.THRESH_BINARY_INV)[1]
    #for use with black background
    thresh = cv2.threshold(blurred, 43, 255, cv2.THRESH_BINARY)[1]
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
                newY = min(len(img)-1,max(0,cY+j))
                pixel = img[newY][newX]
                r += pixel[0]
                g += pixel[1]
                b += pixel[2]
                h_curr,s_curr,v_curr = colorsys.rgb_to_hsv(b/255, g/255, r/255)
                #pixel_hsv = img_hsv[newY][newX]
                #h += pixel_hsv[0]
                #s += pixel_hsv[1]
                #v += pixel_hsv[2]
                h += h_curr
                s += s_curr
                v += v_curr
        r /= 400
        g /= 400
        b /= 400
        h /= 400
        s /= 400
        v /= 400
        #h /= (400*179)
        #s /= (400*255) # was 360
        #v /= (400*255) # was 360

        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        area = cv2.contourArea(c)
        rgb_val, name = get_HSVcolor(h,s,v)
        if name == color and area > 2000:
            center, params, rotation = cv2.minAreaRect(c)
            real_y = -.39316*cX-18.2#-.4068*cX -19#- 14.0149 #-.62*cX-26.44 #-.682 - NEED TO RECALIBRATE THIS
            real_x = -.3779*cY+284#-.389*cY + 280#278.343 #-.52*cY+332.238 #.49
            #arm.set_joint_angle({6:0})
            arm.set_tool_pose(real_x,real_y,60)
            arm.set_tool_pose(real_x,real_y,30)
            time.sleep(1)
            arm.set_tool_pose(real_x,real_y,17.2, speed=25)
            arm.pump_suction()
            arm.set_tool_pose(real_x,real_y, 60)
            arm.set_tool_pose(location[0], location[1], 60) # NOTE CAN CHANGE LAST VALUE TO 10 if we don't want to use the different depth values 
            #if rotation > 20 and 90 - rotation > 20: 
                #arm.set_joint_angle({6:90-rotation})
            arm.set_tool_pose(location[0], location[1], 19) # NOTE CAN CHANGE LAST VALUE TO 10 if we don't want to use the different depth values 
            arm.pump_off()
            arm.set_tool_pose(location[0], location[1], 60)

            cv2.putText(image, name, (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(image, (cX, cY), 7, (int(r), int(g), int(b)), -1)
            
            print("Y: ", cY, "X:", cX)
            #print("Center: ", center)
            #print("Width, height: ", params)
            #print("Angle of rotation: ", rotation)
            #print("RGB: ", [r,g,b])
            #print("HSV: ", [h,s,v])
            # show the image
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            break


def main(file):
    arm.home()
    arm.set_tool_type(WlkataMirobotTool.SUCTION_CUP)
    arm.set_speed(500)
    
    img = pull_from_file(file) 
    output = downsample(img,4,4)
    cv2.imshow("downsampled", output)

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
    time.sleep(5)

    try:
        for i in range(4):
            for j in range(4):
                time.sleep(2)

                [r,g,b] = output[i][j]
                h,s,v = colorsys.rgb_to_hsv(b/255, g/255, r/255)
                rgb, name = get_HSVcolor(h,s,v)

                color = name
                if color == "background_white" or color == "background_black":
                    continue
                print("SEARCHING FOR...", color)
                print("INDEX: ", i, ", ", j)
                end_location = end_locations[i][j]

               
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                color_image = np.asanyarray(color_frame.get_data())

                print("before block picking:", i,j)
                block_picking(color_image, color, end_location)
                print("after block picking:", i,j)

    finally:

        # Stop streaming
        pipeline.stop()

if __name__ == '__main__':
    main("images/pumpkin.png")