import cv2
import os
import shutil

gestures_path = 'imageTreatment/fake_hands_gScale/'

gestures = os.listdir(path)[1:]

dest_test = os.getcwd() + "/input/test/" 
dest_train = os.getcwd() + "/input/train/"

if not os.path.exists(os.getcwd() + "/input"):
    os.mkdir(os.getcwd() + "/input")

if not os.path.exists(dest_test):
    os.makedirs(dest_test)

if not os.path.exists(dest_train):
    os.makedirs(dest_train)

for gesture in gestures:
    path, dirs, files = next(os.walk(os.getcwd() + "/imageTreatment/fake_hands_gScale/" + gestures[gesture]))
    
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