import cv2
import os
import shutil

gestures = {
    0: "fist",
    1: "index",
    2: "pinky",
    3: "l",
    4: "two",
    5: "three",
    6: "ronaldinho",
    7: "metal",
    8: "palm",
    9: "jesus"
}
 
dest_test = os.getcwd() + "/input/test/" 
dest_train = os.getcwd() + "/input/train/"

if not os.path.exists(os.getcwd() + "/input"):
    os.mkdir(os.getcwd() + "/input")

if not os.path.exists(dest_test):
    os.makedirs(dest_test)

if not os.path.exists(dest_train):
    os.makedirs(dest_train)

for gesture in gestures:
    path, dirs, files = next(os.walk(os.getcwd() + "/poses/" + gestures[gesture]))
    
    file_count = len(files) 
    train =round(0.7 * file_count)

    test = 0
    for file in files:
        src = path + "/" + file
        if(test) < train:
            shutil.copy(src, dest_train + "/" + str(gesture) + "_"  + str(test) + ".png")
        else:
            shutil.copy(src, dest_test + "/" + str(gesture) + "_"  + str(test-train)+ ".png")
        test += 1