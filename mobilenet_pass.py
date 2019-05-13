from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.models import Model
import os
import pandas as pd
import numpy as np

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
dfh = hands.loc[:, 'poseID':'T5']
df_col = pd.concat([dfh,df], axis=1)

print(df_col)

df_col.to_csv(os.getcwd() + "/data/fake-hands2.csv")