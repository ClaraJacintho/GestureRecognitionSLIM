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


model = Ridge(alpha=0.0001, copy_X=True, fit_intercept=True, max_iter=None, normalize=True, 
              random_state=False, solver='auto', tol=0.01)
model.fit(X, y)


#----------------------------------------------------------------------------------------------------------------------------------
'''
y_actual = y_test
y_predicted = model.predict(X_test) 

rms = sqrt(mean_squared_error(y_actual, y_predicted))
print(rms)
print(y_actual[:10])
print(np.around(y_predicted[:10], decimals=1))
'''

data = pd.read_csv('data/real-hands.csv',  index_col=False)



X_rh = data.ix[:,V_START_COL:V_END_COL].as_matrix()
y_predicted_rh = model.predict(X_rh) 
print(np.around(y_predicted_rh, decimals=3))


#-------------------------------------------------------------------------------------------------------------------
'''
new_tendons = model.predict(X_all) 
new_tendons.shape # this should match above

# new_tendons now needs to replace the columns from START_COL to END_COL
data_old = pd.read_csv('../data/hand_data3_separated.csv',  index_col=False) +

#modify the data
data_new = data_old
columns = ["T1", "T2", "T3", "T4", "T5"]
for i, col in enumerate(columns):    
    print(i, col)
    data_new = data_new.drop([col], axis=1)
    data_new.insert(loc=i+3, column=col, value=new_tendons[:,i],)
    
data_new

#write new tedoncs to csv file
data_new.to_csv(path_or_buf='../data/hand_data3_mirror.csv', index=False)
'''