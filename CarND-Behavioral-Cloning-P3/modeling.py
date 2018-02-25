import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D
from keras.optimizers import Adam
from random import shuffle
from sklearn.model_selection import train_test_split
import sklearn
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings("ignore")
path = "../Data/driving_log.csv"

def getSamples():
    samples = []
    #with open('data/driving_log.csv') as csvfile:
    #    reader = csv.reader(csvfile)
    #    for line in reader:
    #        samples.append(line)
    with open('data/new_img/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    shuffle(samples)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples

#Keras model derived from the Nvidia report 
def nvidiaModel():
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - 1, input_shape=(160, 320, 3)))
    #Remove top 50 pixels and the bottom 25 pixels. Don't drop sides.
    model.add(Cropping2D(cropping=((50, 25), (0,0))))
    model.add(Convolution2D(24, 5, strides=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, strides=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, activation='relu'))
    model.add(Convolution2D(64, 3, activation='relu'))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    opt = Adam(lr=0.0001)
    model.compile(loss='mse', optimizer=opt)

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
                img, angle = imageProcessing(batch_sample)           
                images.append(img)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def imageProcessing(sample, cutoff= 0.33):
    angle = float(sample[3])

    #To remove zero angle bias
   # if(abs(angle) < 0.05):
   #     cutoff = 0.1
    mid_cutoff =  (1-cutoff)/2 + cutoff

    # Randomly pick between left, right, and center image with weighting to handle
    # zero angle bias.
    pick_camera = np.random.uniform()
    img_path = None

    #Use the center image
    if pick_camera <= cutoff:
        img_path = './data/IMG/' + sample[0].split("\\")[-1]
    # Use the left image
    elif pick_camera > cutoff and pick_camera <= mid_cutoff:
        img_path = './data/IMG/' + sample[1].split("\\")[-1]
        angle += 0.25
    # Use the right image
    else:
        img_path = './data/IMG/' + sample[2].split("\\")[-1]
        angle += -0.25

    img = cv2.imread(img_path)

    pick_flip = np.random.randint(2)
    if pick_flip == 1:
        img = cv2.flip(img, 0)
        angle *= -1

    pick_bright = np.random.randint(2)
    if pick_bright == 1:
        img = augmentBrightness(img)
    return img, angle

def augmentBrightness(img):
    img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img_hsv = np.array(img_hsv, dtype = np.float64)
    rand_bright = 2 * np.random.uniform()
    img_hsv[:,:,2] = img_hsv[:, :, 2] * rand_bright
    img_hsv[:,:,2][img_hsv[:,:,2]>255]  = 255
    img_hsv = np.array(img_hsv, dtype = np.uint8)
    img_aug = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB)
    return img

if __name__ == "__main__":
    train_samples, valid_samples = getSamples()
    train_generator = generator(train_samples, batch_size=128)
    valid_generator = generator(valid_samples, batch_size=128)
    train_steps = (len(train_samples) // 128) + 1 
    valid_steps = (len(valid_samples) // 128) + 1 
    model = nvidiaModel()
    model.fit_generator(train_generator, steps_per_epoch=train_steps, validation_data=valid_generator, validation_steps = valid_steps, epochs=9)
    # print(model.evaluate_generator(validation_samples, steps=3))
    model.save('model7.h5')

# Small generator to test generator mechanics
def smallGenerator(samples, batch_size=32):
    samples = samples[0:72]
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/' + batch_sample[0].split("\\")[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

#A test of 3 images to determine if a model can learn a simple sample size.
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

# Test straightness of samples
def straightness():
    samples = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    straight = 0
    turn = 0
    for sample in samples:
        if float(sample[3]) == 0.0:
            straight += 1
        else:
            turn += 1
    print("Samples include {} straight images and {} turn images".format(staight, turn))

#Show the different augmented images
def plotSamples(samples):
    fig = plt.figure(figsize=(15,15))
    gs_all = gridspec.GridSpec(9, 5)
    for sample in samples:
        img, ang = imageProcessing(sample)
        gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_all[i])
        ax1 = plt.subplot(gs[0])
        ax1.imshow(img)
        ax1.xlabel(str(ang))
    plt.show()
