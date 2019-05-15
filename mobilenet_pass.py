from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from sklearn.model_selection import train_test_split
from keras.models import Model
import os
import pandas as pd
import numpy as np
import re

def read_raw_data():
    # To use mirror data, change the data to `hand_data3_mirror.csv`
    data = pd.read_csv(os.getcwd() + "/data/fake-hands2.csv",  index_col=False) 

    # remove punctuation
    data['desc_list'] = data.description.apply(lambda x: [i for i in re.sub(r'[^\w\s]','',str(x)).lower().split()])
    data['desc_str'] = data.desc_list.apply(lambda x: ' '.join(x))

    # add one-hot encoding for the camera
    camera_data = pd.get_dummies(data.camera_angle)
    data = pd.concat([data, camera_data], axis=1)
    cols = data.columns.tolist()
    cols = cols[:8] + cols[-4:] + cols[8:-4]
    data = data[cols]
    
    # get words and vocabs
    words = [y for x in data.desc_list for y in x]
    vocab = list(set(words))
    # print('number of unique words in our data:', len(vocab), '\nnumber of word tokens in our data: ', len(words))
    
    return data, words, vocab

def create_df():
    df = pd.DataFrame(columns=["f1"])
    for i in range(2, 1001):
        df['f'+str(i)] = -1
    return df

df = create_df()

model = MobileNet()
processed_files = 0
# load and preprocess image
path, dirs, files = next(os.walk(os.getcwd() + "/input/fake_hands"))
for file in files:
    #Image preprocessing - Grayscale
    img_path = path + "/" + file
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # predict the class probabilities
    preds = model.predict(x)
    data = preds.flatten()

    df.loc[len(df)] = data
    print(processed_files)
    processed_files += 1

print(str(processed_files) + " processed files")

hands = pd.read_csv('data/hand_data3_separated.csv',  index_col=False)
hands = hands[:972]
dfh = hands.loc[:, 'poseID':'T5']
df_col = pd.concat([dfh,df], axis=1)

print(df_col)

df_col.to_csv(os.getcwd() + "/data/fake-hands2.csv")
'

data, words, vocab = read_raw_data()

train_data, test_data = train_test_split(data, test_size=0.2)
test_data, heldout_data = train_test_split(test_data, test_size=0.5)

#To use mirror data, change these data to `...data_mirror`
train_data.to_pickle(os.getcwd() +  "/data/train_data4_separated.pkl")
test_data.to_pickle(os.getcwd() + "/data/test_data4_separated.pkl")
heldout_data.to_pickle(os.getcwd() +"/data/heldout_data4_separated.pkl")

print(train_data.shape)
print(test_data.shape)
print(heldout_data.shape)