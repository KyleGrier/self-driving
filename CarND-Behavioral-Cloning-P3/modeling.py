import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D
from random import shuffle
from sklearn.model_selection import train_test_split
import sklearn
import sys
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

path = "../Data/driving_log.csv"

def getSamples():
    samples = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples

def nvidiaModel():
    ch, row, col = 3, 80, 320  # Trimmed image format

    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - 1.,\
        input_shape=(ch, row, col),\
        output_shape=(ch, row, col)))
    #model.add(Lambda(x: x/127.5 - 1, input_shape=(160, 320, 3)))
    model.add(Convolution2D(24, 5, strides=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, strides=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, activation='relu'))
    model.add(Convolution2D(64, 3, activation='relu'))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.compile(loss='mse', optimizer='adam')

    return model

def modelTrainGen(validation_split, epochs):
    train_samples, validation_samples = get_samples()
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)
    model = nvidiaModel()
    model.fit_generator(train_generator, samples_per_epoch = \
                len(train_samples), validation_data=validation_generator, \
                nb_val_samples=len(validation_samples), nb_epoch=3)
    model.save('model.h5')

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def smallGenerator(samples, batch_size=32):
    samples = samples=[:72]
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/' + batch_sample[0].split('/')[-1]
                print(name)
                break
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            break
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
        break
    yield 1

def testModel(model):
    ex_neg = "data/IMG/center_2017_11_18_16_00_07_045.jpg"
    ex_neg_lab = -0.3743682
    ex_zero = "data/IMG/center_2017_11_18_16_00_13_544.jpg"
    ex_zero_lab = 0
    ex_pos = "data/IMG/center_2017_11_18_16_00_14_701.jpg"
    ex_pos_lab = 0.3366786
    ex = [ex_neg, ex_zero, ex_pos]
    X = np.array([plt.imread(img) for img in ex])
    y = np.array([ex_neg_lab, ex_zero_lab, ex_pos_lab])
    model.fit(X, y, epoch=100)
    score = model.evaluate(X, y)
    print(score)

if __name__ == "__main__":
    #for arg in sys.argv[1:]:
    #    print arg
    #print(sys.argv[0])
    #modelTrainGen(X_train, y_train, validation_split, epochs)
    train_samples, validation_samples = getSamples()
    generator = smallGenerator(train_samples)
    #model = nvidiaModel()
    #model.fit_generator(generator, epochs=50)
    #print(model.evaluate_generator(generator))
    model.save('model.h5')
