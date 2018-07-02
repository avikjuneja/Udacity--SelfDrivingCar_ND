import os
import csv
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import sklearn

samples = []

## read out the training data from the csv with image file paths to store in a 'samples' data structure
## each row in csv contains 
##   1. image paths (center, left and right view angles)
##   2. steering angle
##   3. throttle value
##   4. brake amount
##   5. drive speed
def read_dataset(directory):
    with open(directory+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader: 
            for angle in range(3):
                path = line[angle]
                filename = path.split('/')[-1]
                filename = filename.split('\\')[-1]
                path = directory+'IMG/'+filename
                line[angle] = path
            samples.append(line)
            
## creates batches of training data by fetching training image and performing data augmentation by:
## adding data for center, left and right angles
## flipping training images over y-axis
## converting to RGB format, as the inferencing data will be in RGB format

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            measurements = []
            
            for batch_sample in batch_samples:
                if(batch_sample[3] == 'steering'):
                    continue  ## ignore the header in the file
                
                steering = float(batch_sample[3]) ## store steering angle for centered view
                
                ## correction factor for left and right angles
                correction = 0.1 # this is a parameter to tune
                
                for angle in range(3):
                    
                    path = batch_sample[angle]

                    image = cv2.imread(path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    ## convert to RGB to match inferencing data

                    images.append(image)

                # create adjusted steering measurements for the side camera images
                    if(angle == 1):   ## if left angle image add correction
                        steering += correction
                    elif (angle == 2):   ## if right angle image add correction
                        steering -= correction

                    measurements.append(steering)

                    ## data augmentation by flipping image over y-axis (
                    ## gives perception of driving in opposite direction on the track
                    image_flipped = image.copy()
                    image_flipped = cv2.flip(image,1)
                    measurement_flipped = -steering
                    images.append(image_flipped)
                    measurements.append(measurement_flipped)
                


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
#             print('X_train: ',len(X_train))
#             print('Y_train: ',len(y_train))
            
            yield sklearn.utils.shuffle(X_train, y_train)
        
        
# X_train = np.array(images)
# y_train = np.array(measurements)

read_dataset('data_custom/')
shuffle(samples)

## split data for validation and training
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# print(len(train_samples))
# print(len(validation_samples))

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)


from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Activation, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

## Use NVIDIA drive network
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))  ## normalization step
model.add(Cropping2D(cropping=((50,20),(0,0))))    ## crop image to clip background scenery and hood of the car to focus on road
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
#model.add(Dropout(0.5))
model.add(Dense(1))   ## output steering angle

model.compile(loss='mse', optimizer='adam')   ## mean square error since it's a regression and not classification model
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')
