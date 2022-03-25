import cv2
import numpy as np

img = cv2.imread('images/dog.jpg')
dim = (10,10)

# nearest neighbor interpolation looks best
# will abstract to a function later when we have more code
resized = cv2.resize(img, dim, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

cv2.imshow('image',resized)
cv2.waitKey(0)
cv2.destroyAllWindows()