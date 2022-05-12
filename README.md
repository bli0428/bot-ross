# bot-ross
* Calibration: Calibration photo of 16 testing dots is taken with calibration.py. Pixel coordinates for each dot can be found with calibration_pixel.py. The pixel coordinates can be fed into https://docs.google.com/spreadsheets/d/14Ubk51fODmB3k1Oc8sBgr4nkdh_LD6o0SHjEctrt9kI/edit#gid=0 which performs the linear regression to obtain the final mapping from pixel to real-world Mirobot coordinates. 
* Running the progam: Just run main.py and at the bottom specify which image you which to choose
* images: This folder corresponds to input images to reconstruct
* output_images: This folder contains earlier output images from testing contour detection and color mapping as well as the calibration images used on various days. 
* older_code: This folder contains earlier versions of our code that test specific components (color mapping, frame capturing, center detection etc.) which are not needed for main.py to run. 
