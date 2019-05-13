from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import os
import pandas as pd
import numpy as np

def create_df():
    df = pd.DataFrame(columns=["pose"])
    for i in range(1, 1001):
        df['f'+str(i)] = -1
    return df

gestures = {
    1: "fist",
    2: "index",
    3: "pinky",
    4: "l",
    5: "two",
    6: "three",
    7: "ronaldinho",
    8: "metal",
    9: "palm",
    10: "jesus"
}

df = create_df()
print(df['f999'])

base_model = VGG19(weights='imagenet')
model = base_model #Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

processed_files = 0
# load and preprocess image
path, dirs, files = next(os.walk(os.getcwd() + "/input/train"))
for file in files:
    if file == ".DS_Store":
        continue   
    img_path = path + "/" + file
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    pose = int(file[0])

    # predict the class probabilities
    preds = model.predict(x)
    p = preds.flatten()

    data = np.insert(p, 0, pose) # I hate numpy!
    df.loc[len(df)] = data
    print(processed_files)
    processed_files += 1

print(str(processed_files) + " processed files")
print(df.shape)
print(df)

df.to_csv(os.getcwd() + "/data/real-hands.csv")