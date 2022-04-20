import cv2
import numpy as np

def pull_from_file(filepath):
    return cv2.imread(filepath)

def downsample(img, width, length):
    # nearest neighbor interpolation looks best
    return cv2.resize(img, (length, width), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

if __name__ == '__main__':
    img = pull_from_file("images/srinath.jpeg")
    output = downsample(img, 10, 10)
    cv2.imshow("Image", output)
    cv2.imwrite("images/srinath_downsampled2.jpg", output)
