import csv
import os
import sys
import time

import cv2
import keras.backend
import matplotlib
import numpy as np
from keras.layers import Flatten, Dense, Dropout, Lambda, Convolution2D, MaxPooling2D
from keras.models import Sequential

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
        image_center = cv2.imread(os.path.join(dir, 'IMG/' + csvline[0].split('/')[-1]))
        image_left = cv2.imread(os.path.join(dir, 'IMG/' + csvline[1].split('/')[-1]))
        image_right = cv2.imread(os.path.join(dir, 'IMG/' + csvline[2].split('/')[-1]))
        if image_center is None or image_left is None or image_right is None:
            sys.exit("image is None")

        correction = 0.2
        steering_center = float(csvline[3])
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        X.extend((image_center, image_left, image_right))
        y.extend((steering_center, steering_left, steering_right))

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
    Convolution2D(12, 5, 5, activation='relu', init='he_normal'),
    MaxPooling2D(),
    Convolution2D(25, 5, 5, activation='relu', init='he_normal'),
    MaxPooling2D(),
    Flatten(),
    Dense(180, activation='relu', init='he_normal'),
    Dropout(0.5),
    Dense(100, activation='relu', init='he_normal'),
    Dropout(0.5),
    Dense(1, init='he_normal')
])
model.compile(loss='mse', optimizer='adam')
history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
model.save('model.h5')

# show history plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('history.png')

# try to avoid 'NoneType' object has no attribute 'TF_DeleteStatus' error
keras.backend.clear_session()
