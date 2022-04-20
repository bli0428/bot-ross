import cv2
import numpy as np
from tryHSV import get_HSVcolor
import colorsys

def pull_from_file(filepath):
    return cv2.imread(filepath)

def downsample(img, width, length):
    # nearest neighbor interpolation looks best
    return cv2.resize(img, (length, width), fx=0.5, fy=0.5)

if __name__ == '__main__':
    img = pull_from_file("images/pink_flower.jpeg")
    output = downsample(img,10,10)
    cv2.imshow("Image", output)
    cv2.waitKey(0)
    new_output = []
    for i in range(len(output)):
        for j in range(len(output[0])):
            [r,g,b] = output[i][j]
            h,s,v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            rgb, name = get_HSVcolor(h,s,v)
            new_output.append([np.uint8(rgb[0]),np.uint8(rgb[1]),np.uint8(rgb[2])])
    new_output = np.array(new_output)
    new_output = new_output.reshape((10,10,3))
    print(output.shape, output.dtype)
    print(new_output.shape, new_output.dtype)
    cv2.imshow("Converted", new_output)
    cv2.imwrite("converted_flower.jpg", new_output)
