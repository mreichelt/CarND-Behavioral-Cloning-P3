import csv
import cv2
import os
import sys
import time

import numpy as np

import keras.backend
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Activation, Lambda, Convolution2D, MaxPooling2D
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


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
        center = 0
        left = 1
        right = 2
        image = cv2.imread(os.path.join(dir, 'IMG/' + csvline[center].split('/')[-1]))
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


def flip_images(X, y):
    """Flips the images + steering angle, returns new arrays"""
    X = np.copy(X)
    y = np.copy(y) * -1.0
    for i, x in enumerate(X):
        X[i] = np.fliplr(x)
    return X, y


# Print some stuff about the data
t = time.process_time()
X_train, y_train = load_all_driving_data()
t = time.process_time() - t
print('data loaded in {:.2f}s'.format(t))
print('X_train shape is {}'.format(X_train.shape))
print('y_train shape is {}'.format(y_train.shape))

# Augment the data
X_flip, y_flip = flip_images(X_train, y_train)
X_train = np.concatenate((X_train, X_flip))
y_train = np.concatenate((y_train, y_flip))

# Now let's train!
input_shape = X_train.shape[1:]
model = Sequential([
    Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape),
    Convolution2D(16, 5, 5, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(loss='mse', optimizer='adam')
history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1)
model.save('model.h5')

# show history plot
print(history.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('history.png')

# try to avoid 'NoneType' object has no attribute 'TF_DeleteStatus' error
keras.backend.clear_session()
