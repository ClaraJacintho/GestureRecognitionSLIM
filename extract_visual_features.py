from keras.applications.vgg19 import VGG19
from keras.applications import MobileNet
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from sklearn.model_selection import train_test_split
from progress.bar import ChargingBar
import cv2
import os
import pandas as pd
import numpy as np
import argparse
import re
import gesture_utils as gu

parser = argparse.ArgumentParser(description='Train the necessary models')
parser.add_argument(
    '-v',
    '--vgg19',
    action='store_true',
    help='Train the visual model with VGG19. Default trains with MobileNet')
parser.add_argument(
    '-s',
    '--synthetic_hands',
    action='store_true',
    help='Train the visual model for the synthetic hands. Default trains with real hands')
parser.add_argument(
    '-g',
    '--grayscale',
    action='store_true',
    help='Train with the images in grayscale. Default trains with RGB')

args = parser.parse_args()

if args.synthetic_hands:
    hand_path = os.getcwd() + "/input/synthetic_hands"
    target_path = os.getcwd() + "/data/synthetic-hand" 
    print("From synthetic hands")
else:
    hand_path = os.getcwd() + "/input/all_hands"
    print("From real hands")
    target_path = os.getcwd() + "/data/real-hands" 

if args.grayscale:
    target_path = target_path + "-grayscale"
    print("In grayscale")

if args.vgg19:
    model = VGG19(weights='imagenet')
    target_path = target_path + "-vgg19.csv"
    print("Using VGG19")
else:
    model = MobileNet()
    target_path = target_path + "-mn.csv"
    print("Using MobileNet")



df = gu.create_df(args.synthetic_hands)

# load and preprocess image
path, dirs, files = next(os.walk(hand_path))
processed_files = 0
bar = ChargingBar('Processing', max=len(files))
for file in files:
    img_path = path + "/" + file

    #preprocess it
    picture = cv2.imread(img_path)
    img = cv2.resize(picture,(224, 224))

    if args.grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.stack((img,)*3, axis=-1) #models expect images with 3 channels
    else:
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #conversion openCV -> PIL

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # predict the class probabilities - extracting the visual features
    preds = model.predict(x)
    p = preds.flatten()
    if args.synthetic_hands:
        df.loc[len(df)] = p
    else:
        pose = int(file[0])
        data = np.insert(p, 0, pose)
        data = np.insert(data, 1, 0)
        df.loc[len(df)] = data
        df["camera_angle"] = "straight_on"

    processed_files += 1
    bar.next()

bar.finish()
if args.synthetic_hands:
    hands = pd.read_csv('input/hand_data3_separated.csv',  index_col=False)
    hands = hands[:972]
    dfh = hands.loc[:, 'poseID':'T5']
    df = pd.concat([dfh,df], axis=1)

print("Processed " + str(processed_files) + " files")

print("Saved results at " + target_path)
df.to_csv(target_path, index=False)

if args.synthetic_hands:
    data, _, _ = gu.process_raw_data(df)
    train_data, test_data = train_test_split(data, test_size=0.2)
    test_data, heldout_data = train_test_split(test_data, test_size=0.5)

    pickle_path = target_path[:-4] 
    print("Pickled files at " + pickle_path)

    train_data.to_pickle(pickle_path + "-train.pkl")
    test_data.to_pickle(pickle_path + "-test.pkl" )
    heldout_data.to_pickle(pickle_path+ "-heldout.pkl" )