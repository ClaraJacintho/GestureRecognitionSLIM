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
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle
import gesture_utils as gu
import argparse

parser = argparse.ArgumentParser(description='Train the necessary models')
parser.add_argument(
    '-v',
    '--vgg19',
    action='store_true',
    help='Train the visual model with VGG19. Default trains with MobileNet')
parser.add_argument(
    '-g',
    '--grayscale',
    action='store_true',
    help='Train with the images in grayscale. Default trains with RGB')

args = parser.parse_args()

if args.vgg19:
       train_path = "data/synthetic-hand-vgg19"
       test_path = "data/synthetic-hand-vgg19"
else:
        train_path ="data/synthetic-hand-mn"
        test_path =  "data/synthetic-hand-mn"

if args.grayscale:
        train_path = train_path + "-grayscale-train.pkl" 
        test_path = test_path + "-grayscale-test.pkl"
else:
        train_path = train_path + "-train.pkl" 
        test_path = test_path + "-test.pkl"

train_data, test_data = gu.read_in_data(train_path, test_path)
train_data[:10]

START_COL = 'T1' 
END_COL = 'T5'
V_START_COL = 'f1'
V_END_COL = 'f1000'

y = train_data.ix[:,START_COL:END_COL].as_matrix()
X = train_data.ix[:,V_START_COL:V_END_COL].as_matrix()
y_test = test_data.ix[:,START_COL:END_COL].as_matrix()
X_test = test_data.ix[:,V_START_COL:V_END_COL].as_matrix()


print("train", X.shape, y.shape)
print("test", X_test.shape, y_test.shape)



model = Ridge(alpha=0.001, copy_X=True, fit_intercept=True, max_iter=None, normalize=True, 
              random_state=False, solver='auto', tol=0.01)
model.fit(X, y)


#----------------------------------------------------------------------------------------------------------------------------------


y_actual = y_test
y_predicted = model.predict(X_test) 

rms = sqrt(mean_squared_error(y_actual, y_predicted))
print("Mean squared error is :", rms)
print("Actual tendons:")
print(y_actual[:10])

print("Predicted tendons are :")
print(np.around(y_predicted[:10], decimals=1))

if args.vgg19:
        hand_path = "data/real-hands-vgg19"
else:
        hand_path ="data/real-hands-mn"

if args.grayscale:
    hand_path = hand_path + "-grayscale.csv"
else:
    hand_path = hand_path + ".csv"




data = pd.read_csv(hand_path,  index_col=False)

X_rh = data.ix[:,V_START_COL:V_END_COL].as_matrix()
pose_and_camera = data.ix[:,'poseID':'camera_angle'].as_matrix()

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
partial_result = np.concatenate((pose_and_camera, y_predicted_rh), axis = 1)

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

hand_path = hand_path[:-3] + "-tendon.csv"
df.to_csv(hand_path)