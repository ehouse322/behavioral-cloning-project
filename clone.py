import csv
import cv2
import numpy as np

lines = []

with open('data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

images = []
measurements = []

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

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25),(0,0))))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
# model.add(Dropout(0.7))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
# model.add(Dropout(0.7))
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
