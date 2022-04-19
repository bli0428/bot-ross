import colorsys
from sre_constants import GROUPREF_EXISTS
import cv2
import numpy as np
import imutils
# 1. Establish range of H values for colors: red, orange, yellow, blue, green purple, light background (white), dark background (black)
# Create a default rgb tuple for each of those colors for visualization purposes.
names = ["red","orange","yellow","green","blue","purple","background_white", "background_black"]
default_rgb_values = [[254,231,31],[254,142,31], [241,33,17],[25,171,37],[16,119,161],[184,55,185],[255,255,255],[0,0,0]]
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
		


# 2. import one of the output camera files ("original_image...")
img = cv2.imread("original_image_6.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray[0][0])

cv2.imshow("b/w", gray)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#for use with non-black background
#thresh = cv2.threshold(blurred, 1.86*gray[0][0]-237.87, 255, cv2.THRESH_BINARY_INV)[1]
#for use with black background
thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)[1]
#second value should depend on overall brightness

cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

image = img

img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

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
	h /= 400*180 # ALSO DIVIDING THE HSV VALUES BY 180 bc cv2.cvtcolor (we belive) returns int not float
	s /= 400*255
	v /= 400*255
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.circle(image, (cX, cY), 7, (int(r), int(g), int(b)), -1)
	rgb_val, name = get_HSVcolor(h,s,v)
	#cv2.putText(image, "center", (cX - 20, cY - 20),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	cv2.putText(image, name, (cX - 20, cY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	# show the image
	cv2.imshow("Image", image)
	cv2.waitKey(0)

img = img.reshape((307200,3))


# 3. convert file to np array of rgb values
# - convert rgb values to hsv values

#4. loop through all pixels and store (1) array with closest color name based on H range for pixel
#  (2) array with the corresponding default rgb tuple for pixel 
# output = []
# colors = []
# for i in range(len(img)):
#     pixel = img[i]
#     hsv = colorsys.rgb_to_hsv(pixel[0]/255,pixel[1]/255,pixel[2]/255)
#     if hsv[1] < .25:
#         output.append(default_rgb_values[6])
#         colors.append(names[6])
#     elif hsv[2] < .25:
#         output.append(default_rgb_values[7])
#         colors.append(names[7])
#     else:
#         if hsv[0] <= .033 or hsv[0] >= .9055:
#             output.append(default_rgb_values[0])
#             colors.append(names[0])
#         elif .033 < hsv[0] <= .1194 :
#             output.append(default_rgb_values[1])
#             colors.append(names[1])
#         elif .1194 < hsv[0] <= .1833 :
#             output.append(default_rgb_values[2])
#             colors.append(names[2])
#         elif .1833 < hsv[0] <= .46 :
#             output.append(default_rgb_values[3])
#             colors.append(names[3])
#         elif .46 < hsv[0] <= .7388 :
#             output.append(default_rgb_values[4])
#             colors.append(names[4])
#         else:
#             output.append(default_rgb_values[5])
#             colors.append(names[5])

# print(output)
# output = np.array(output)
# output = np.reshape(output, (480,640,3))
# print(output.shape)


#cv2.imshow(output)
#cv2.imwrite("output6.jpeg",output)


#5. reshape array of default rgb tuples to image size (480,640,3) and save as image for visualization