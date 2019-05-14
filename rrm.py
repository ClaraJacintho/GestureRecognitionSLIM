import pandas as pd
from sklearn.metrics import *
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from collections import defaultdict as dd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from slir import SparseLinearRegression

from sklearn.metrics import mean_squared_error
from math import sqrt
import math

import operator
import re
from collections import Counter

#from textblob import TextBlob
#from textblob import Word

from IPython.display import HTML, display
from IPython.display import Image

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle

def read_raw_data():
    data = pd.read_csv('data/hand_data3_separated.csv',  index_col=False)

    # remove punctuation
    data['desc_list'] = data.description.apply(lambda x: [i for i in re.sub(r'[^\w\s]','',str(x)).lower().split()])
    data['desc_str'] = data.desc_list.apply(lambda x: ' '.join(x))

    #add one-hot encoding
    camera_data = pd.get_dummies(data.camera_angle)
    data = pd.concat([data, camera_data], axis=1)
    cols = data.columns.tolist()
    cols = cols[:8] + cols[-4:] + cols[8:-4]
    data = data[cols]
    
    #get words and vocabs
    words = [y for x in data.desc_list for y in x]
    vocab = list(set(words))
    print('number of unique words in our data:', len(vocab), '\nnumber of word tokens in our data: ', len(words))
    
    return data, words, vocab

def read_in_data():
    train_data = pd.read_pickle("data/train_data3_separated.pkl")
    test_data = pd.read_pickle("data/test_data3_separated.pkl")
    return train_data, test_data

def stack_training_data(mydata):
    s = mydata.apply(lambda x: pd.Series(x['desc_list']),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'word'
    mydata = mydata.join(s)
    return mydata

def remove_unwated_words(mydata, vocab, words):
    wanted_words = list(set(words))
    unwanted_words = {'hand', 'and', 'the', 'a', 'with', 'is', 'are', 'to', 'of', 'finger', 'fingers', 'thumb'}
    unwanted_tags = {}
    for curr_word in vocab:
        if curr_word in unwanted_words:
            wanted_words.remove(curr_word)
    mydata = mydata.loc[mydata['word'].isin(wanted_words)]
    return mydata

_, words, vocab = read_raw_data()
train_data, test_data = read_in_data()
train_data[:10]

data, words, vocab = read_raw_data()

data.columns[:15]

START_COL = 'T1' 
END_COL = 'T5'
V_START_COL = 'f1'
V_END_COL = 'f1000'

y = train_data.ix[:,START_COL:END_COL].as_matrix()
X = train_data.ix[:,V_START_COL:V_END_COL].as_matrix()
y_test = test_data.ix[:,START_COL:END_COL].as_matrix()
X_test = test_data.ix[:,V_START_COL:V_END_COL].as_matrix()
y_all = data.ix[:,START_COL:END_COL].as_matrix()
X_all = data.ix[:,V_START_COL:V_END_COL].as_matrix()

print("train", X.shape, y.shape)
print("test", X_test.shape, y_test.shape)
print("all", X_all.shape, y_all.shape)


model = Ridge(alpha=0.001, copy_X=True, fit_intercept=True, max_iter=None, normalize=True, 
              random_state=False, solver='auto', tol=0.01)
model.fit(X, y)


#----------------------------------------------------------------------------------------------------------------------------------

y_actual = y_test
y_predicted = model.predict(X_test) 

rms = sqrt(mean_squared_error(y_actual, y_predicted))
print("Mean squared error is :", rms)
print(y_actual[:10])

print("Predicted tendons are :")
print(np.around(y_predicted[:10], decimals=1))

data = pd.read_csv('data/real-hands.csv',  index_col=False)

X_rh = data.ix[:,V_START_COL:V_END_COL].as_matrix()

# Getting the poses to add them in the csv
poses_list = data.ix[:,'pose']
poses_row = np.asarray([poses_list])
poses = poses_row.transpose()

# Getting the tendon values using the model from Boise
y_predicted_rh = model.predict(X_rh)
print("Predicted tendons") 
print(np.around(y_predicted_rh, decimals=1))

# Can be uncommented to check for dimensions in case
# they do not seem to match
"""
print(X_rh.shape)
print(type(X_rh))

print(y_predicted_rh.shape)
print(type(y_predicted_rh))

print(poses.shape)
print(type(poses))
"""

# Adding poses to the tendon values
partial_result = np.concatenate((poses, y_predicted_rh), axis = 1)

# Adding the result to the features
final2 = np.concatenate((partial_result, X_rh), axis = 1)

# Creating the feature list to add as header
feature_list = []
for i in range(1, 1001):
        feature_list.append("f" + str(i))

tendon_list = ["pose", "camera_angle", "T1", "T2", "T3", "T4", "T5"]

# Adding the feature list to the other headers 
tendon_list.extend(feature_list)

# Saving the results in a csv
df = pd.DataFrame(final2)
df.columns = tendon_list
print(df)
df.to_csv("data/real-hands-with-tendons.csv")