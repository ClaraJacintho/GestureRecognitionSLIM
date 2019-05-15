import pandas as pd
import numpy as np
import os
import re

def create_df(synthetic):
    if synthetic:
        df = pd.DataFrame(columns=["f1"])
        for i in range(2, 1001):
            df['f'+str(i)] = -1
    else:
        df = pd.DataFrame(columns=["poseID", "camera_angle"])
        for i in range(1, 1001):
            df['f'+str(i)] = -1
    return df

def process_raw_data(data):

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

def read_in_data():
    train_data = pd.read_pickle("data/train_data3_separated.pkl")
    test_data = pd.read_pickle("data/test_data3_separated.pkl")
    print(train_data)
    return train_data, test_data
