import csv
import cv2
import numpy as np

lines = []

# we first read the data from our driving_log.csv file
with open('data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

images = []
measurements = []

# here we augment the data, adding in the images for the left, right and center cameras. we also convert from BGR to RGB, and add in steering wheel angles with correction for the left and right angles.
for line in lines[1:]:
  for i in range(3):
    source_path = line[i]
    tokens = source_path.split('/')
    filename = tokens[-1]
    local_path = "./data/IMG/" + filename
    bgr_image = cv2.imread(local_path)
    image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    images.append(image)
  correction = 0.25
  measurement = line[3]
  measurements.append(measurement) 
  measurements.append(float(measurement)+correction)
  measurements.append(float(measurement)-correction+0.15)

augmented_images = []
augmented_measurements =[]

# we augment the images by flipping them
for image, measurement in zip(images, measurements):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  flipped_image = cv2.flip(image, 1)
  flipped_measurement = float(measurement) * -1.0
  augmented_images.append(flipped_image)
  augmented_measurements.append(flipped_measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print(X_train.shape)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import plot_model

# we implement our model using keras
model = Sequential()
# before any layers, we normalize the data and crop the images
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25),(0,0))))
# our model consists of two convolution layers with RELU activation and two max pooling layers followed by flatten and dense layers.
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

# we use the adam optimizer, so that we do not need to set the learning rate manually.
model.compile(optimizer='adam', loss='mse')
# we keep 20% of the data for validation and shuffle the data.
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

plot_model(model, to_file='model.png')
model.save('model.h5')
