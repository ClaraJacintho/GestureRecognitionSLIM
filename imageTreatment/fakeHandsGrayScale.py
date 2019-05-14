
import numpy as np
import cv2
import os

"""
#Only in grayScale
#Read Image
img = cv2.imread('screenshots-jpg/pic101straight_on.jpg')
#Display Image
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Applying Grayscale filter to image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Saving filtered image to new file
cv2.imwrite('graytest.jpg', gray)
"""

path = 'screenshots-jpg/'
gestures_path = 'fake_hands_gScale/'

gestures = os.listdir(path)[1:]

pic_no = 0

for i in gestures:
    print(i)
    if i == ".DS_Store":
        continue
	
    
    img = cv2.imread(path + i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pic_no += 1

    save_img = np.array(gray)
    cv2.imwrite(gestures_path + "/" + str(pic_no) + ".jpg", save_img)
    