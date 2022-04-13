# 1. Establish range of H values for colors: red, orange, yellow, blue, green purple, light background (white), dark background (black)
# Create a default rgb tuple for each of those colors for visualization purposes.
names = ["red","orange","yellow","green","blue","purple","background_white", "background_black"]
default_rgb_values = [[254,231,31],[254,142,31], [241,33,17],[25,171,37],[16,119,161],[184,55,185],[255,255,255],[0,0,0]]
# H_ranges = 
# while color ranges can be determined by the hue value, 
# might need to check black/white another way because white seems to be when saturation is like <.25
# and black seems to be when saturation is value is below <.35 


# 2. import one of the output camera files ("original_image...")

# 3. convert file to np array of rgb values
# - convert rgb values to hsv values

#4. loop through all pixels and store (1) array with closest color name based on H range for pixel
#  (2) array with the corresponding default rgb tuple for pixel 

#5. reshape array of default rgb tuples to image size (480,640,3) and save as image for visualization