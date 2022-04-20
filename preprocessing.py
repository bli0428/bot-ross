import cv2
import numpy as np
from tryHSV.py import get_HSVcolor

def pull_from_file(filepath):
    return cv2.imread(filepath)

def downsample(img, width, length):
    # nearest neighbor interpolation looks best
    return cv2.resize(img, (length, width), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

if __name__ == '__main__':
    img = pull_from_file("images/daddy.png")
    output = downsample(img,10,10)
    cv2.imshow("Image", output)
    cv2.waitKey(0)
    new_output = []
    for i in range(len(img)):
        for j in range(len(img[0])):
            [r,g,b] = output[i][j]
            h,s,v = colorsys.rgb_to_hsv(b/255, g/255, r/255)
            rgb, name = get_HSVcolor(h,s,v)
            new_output.append(rgb)
    new_output = np.array(new_output)
    new_output = new_output.reshape((10,10,3))
    cv2.imshow("Converted", new_output)
    cv2.waitKey(0)
