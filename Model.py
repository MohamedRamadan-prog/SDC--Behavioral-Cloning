import csv
import cv2
import numpy as np
import tensorflow
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

lines = []
   
with open('./Output/driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader:
       lines.append(line)
       
images =[]
measurements = []

# this is to edit the current path to fit the floyd path
for line in lines:
   measurement = float(line[3])
   correction = 0.25
   
   # skip it if ~0 speed - not representative of driving behavior
   if float(line[6]) < 0.1 :
           continue
   for i in range(3):
       source_path = line[i]
       filename = source_path.split('\\')[-1]
       current_path = './Output/IMG/' + filename
       image = cv2.imread(current_path)
       images.append(image)
       
       # Center image
       if (i == 0):
           measurements.append(measurement)
       # Left image
       elif(i == 1):
           measurements.append(measurement + correction)
       # Right image
       else:
           measurements.append(measurement - correction)
           
augmented_images, augmented_measurements = [], []

for image , measurement in zip(images,measurements):
   augmented_images.append(image)
   augmented_measurements.append(measurement)
   augmented_images.append(cv2.flip(image,1))
   augmented_measurements.append(-1*measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

def createPreProcessingLayers():
    """
    Creates a model with the initial pre-processing layers.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model

def nVidiaModel():
    """  65
    Creates nVidea Autonomous Car Group model
    """
    model = createPreProcessingLayers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


model=nVidiaModel()
model.compile(loss='mse', optimizer='adam')

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection  import train_test_split    
history_object = model.fit(X_train, y_train, validation_split=0.3, shuffle = True, nb_epoch = 2)
model.save('./model.h5')
print ('Model Saved')

### plot the training and .Kvalidation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('./MSE.pdf')

