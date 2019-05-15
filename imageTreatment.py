
import numpy as np
import cv2
import os

"""
img_path = '0.jpg'
img = cv2.imread(img_path, 0)
blur = cv2.GaussianBlur(img, (11,11), 0)
blur = cv2.medianBlur(blur, 15)
thresh = cv2.threshold(blur,210,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
thresh = cv2.bitwise_not(thresh)
save_img = cv2.resize( thresh, (244,244) )
img = np.array(save_img)
img = img.reshape( (244,244,1) )
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#img = np.array(img)
print(img.shape)



#Only in grayScale
#Read Image
img = cv2.imread('0.jpg')
#Display Image
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#Applying Grayscale filter to image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Saving filtered image to new file
cv2.imwrite('graytest.jpg',gray)
"""


""" 
#-------------
#Canny Edge
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
path = '/Users/delton/Documents/Projet_Industriel/imageTreatment/hand_poses_train/'
gestures_path = '/Users/delton/Documents/Projet_Industriel/imageTreatment/treated_gestures/'
gestures = os.listdir(path)[1:]
pic_no = 0
for i in gestures:
    images = os.listdir(path + i)
    print(images)
    for j in images:
        if j == ".DS_Store":
            continue
        #print(j)
        
        img_path = path + i + '/' + j   
        
        img = cv2.imread(img_path, 0)
        edges = cv2.Canny(img,75,175) #75, 175 = good value
        pic_no += 1
        save_img = np.array(edges)
        cv2.imwrite(gestures_path + "/" + str(pic_no) + ".jpg", save_img)
#----------
"""

"""
#-----
#Background removing
import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('67.jpg')
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (50,50,244,244)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()


#-------------
#Without background and the background white in GrayScale

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

path = 'hand_poses_train/'
gestures_path = 'without_background_images/'

gestures = os.listdir(path)[1:]

pic_no = 0
grayScale = True

for i in gestures:
    images = os.listdir(path + i)
    print(images)
    for j in images:
        if j == ".DS_Store":
            continue
        #print(j)
        
        img_path = path + i + '/' + j   
        
        img = cv2.imread(img_path)
     
        mask = np.zeros(img.shape[:2],np.uint8)

        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (50,50,224,224)

        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img1 = img*mask2[:,:,np.newaxis]


        #Get the background
        background = img - img1

        #Change all pixels in the background that are not black to white
        background[np.where((background > [0,0,0]).all(axis = 2))] =[255,255,255]

        #Add the background and the image
        final = background + img1

        
        pic_no += 1
        if grayScale:
            final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
        
        save_img = np.array(final)
        cv2.imwrite(gestures_path + "/" + str(pic_no) + ".jpg", save_img)
#----------
"""

path, dirs, files = next(os.walk(os.getcwd() + "/input/train"))
for file in files:      
    img_path = path + "/" + file 
    img = cv2.imread(img_path)
    
    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (50,50,224,224)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    
    img1 = img*mask2[:,:,np.newaxis]


    #Get the background
    background = img - img1

    #Change all pixels in the background that are not black to white
    background[np.where((background > [0,0,0]).all(axis = 2))] =[255,255,255]

    #Add the background and the image
    final = background + img1

    final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
