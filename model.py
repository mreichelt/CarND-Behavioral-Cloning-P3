import csv
import os
import sys
import time

import cv2
import keras.backend
import matplotlib
import numpy as np
from keras.layers import Flatten, Dense, Dropout, Lambda, Convolution2D, MaxPooling2D, Cropping2D
from keras.models import Sequential
import sklearn.utils
from sklearn.model_selection import train_test_split

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


def load_single_data_folder(dir, steer_correction=0.2):
    """Adjusts image paths + get steering angles"""
    samples = []
    csvlines = loadcsv(os.path.join(dir, 'driving_log.csv'))
    for csvline in csvlines:
        image_center = os.path.join(dir, 'IMG/' + csvline[0].split('/')[-1])
        image_left = os.path.join(dir, 'IMG/' + csvline[1].split('/')[-1])
        image_right = os.path.join(dir, 'IMG/' + csvline[2].split('/')[-1])

        steering_center = float(csvline[3])
        samples.append([image_center, steering_center])
        samples.append([image_left, (steering_center + steer_correction)])
        samples.append([image_right, (steering_center - steer_correction)])

    return samples


def load_all_samples(root='driving_data'):
    """Loads all driving samples"""
    samples = []
    for dir in os.listdir(root):
        dir = os.path.join(root, dir)
        if os.path.isdir(dir):
            print('loading {}'.format(dir))
            samples.extend(load_single_data_folder(dir))

    return samples


def generator(samples, batch_size=32, add_flipped=True):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample[0])
                angle = batch_sample[1]
                if image is None:
                    sys.exit("image is None")
                images.append(image)
                angles.append(angle)

                if add_flipped:
                    # Augment the data: add flipped image + angle
                    images.append(np.fliplr(image))
                    angles.append(-angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# split samples into train/validation samples (no images loaded yet)
samples = load_all_samples()
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Print some stuff about the data
image_shape = cv2.imread(train_samples[0][0]).shape
print('input: {} training and {} validation samples'.format(len(train_samples), len(validation_samples)))
print('image shape: {}'.format(image_shape))

# Now let's train!
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)
model = Sequential([
    Lambda(lambda x: x / 255.0 - 0.5, input_shape=image_shape),
    Cropping2D(cropping=((70, 25), (0, 0))),
    Convolution2D(24, 3, 3, activation='relu', subsample=(2, 2)),
    Convolution2D(36, 3, 3, activation='relu', subsample=(2, 2)),
    Convolution2D(48, 3, 3, activation='relu', subsample=(2, 2)),
    Convolution2D(64, 3, 3, activation='relu'),
    Convolution2D(64, 3, 3, activation='relu'),
    Flatten(),
    Dropout(0.5),
    Dense(100),
    Dropout(0.5),
    Dense(50),
    Dropout(0.5),
    Dense(10),
    Dense(1)
])
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(
    train_generator,
    samples_per_epoch=len(train_samples) * 2,  # normal + flipped images
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples) * 2,  # normal + flipped images
    nb_epoch=2
)
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
