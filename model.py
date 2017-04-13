import csv
import cv2
import os
import sys
import time

import numpy as np

import keras.backend
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Dropout, Activation


def loadcsv(file):
    """Returns all data of a CSV file"""
    lines = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines


def load_single_data_folder(dir):
    """Loads a driving data folder"""
    X, y = [], []
    csvlines = loadcsv(os.path.join(dir, 'driving_log.csv'))
    for csvline in csvlines:
        # TODO: load images 2 and 3, too!
        image = cv2.imread(os.path.join(dir, 'IMG/' + csvline[0].split('/')[-1]))
        if image is None:
            sys.exit("image is None")
        X.append(image)
        y.append(float(csvline[3]))

    return np.array(X), np.array(y)


def load_all_driving_data(root='driving_data'):
    """Loads all driving data"""
    X_all, y_all = [], []
    for dir in os.listdir(root):
        dir = os.path.join(root, dir)
        if os.path.isdir(dir):
            print('loading {}'.format(dir))
            X, y = load_single_data_folder(dir)
            X_all.extend(X)
            y_all.extend(y)
    return np.array(X_all), np.array(y_all)


t = time.process_time()
X_train, y_train = load_all_driving_data()
t = time.process_time() - t
print('data loaded in {:.2f}s'.format(t))
print('X shape is {}'.format(X_train.shape))
print('y shape is {}'.format(y_train.shape))

# Now let's train!

model = Sequential([
    Flatten(input_shape=X_train.shape[1:]),
    Dense(1)
])
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
model.save('model.h5')

# try to avoid 'NoneType' object has no attribute 'TF_DeleteStatus' error
keras.backend.clear_session()
