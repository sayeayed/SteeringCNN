import os
import csv
import numpy as np
from sklearn.utils import shuffle

## Read in frame data
samples = []
with open('/../opt/carnd_p3/data/driving_log.csv') as csvfile:  #open the log file
    reader = csv.reader(csvfile)  #as a readable csv
    for line in reader:
        samples.append(line)  #add each line of the log file to samples
samples = samples[1:] # to remove table header
samples = shuffle(samples) # shuffle entire sample set before splitting into training and validation so that training isn't biased

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

from scipy import ndimage #because cv2.imread() imports the image as BGR, and we want RGB

## Define generator to handle small portions of images at a time so that training is not as memory-heavy
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
#         shuffle(samples)  #shuffle within the training/validation sets
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
#                 name = './IMG/'+batch_sample[0].split('/')[-1]
                path = '/../opt/carnd_p3/data/IMG/'  #assign the location from which to read images
    
                # read in images from all 3 cameras MAKING SURE TO READ IN AS RGB
                center_image = ndimage.imread(path+batch_sample[0].split('/')[-1])
                left_image = ndimage.imread(path+batch_sample[1].split('/')[-1])
                right_image = ndimage.imread(path+batch_sample[2].split('/')[-1])
#                 center_image = cv2.imread(name)
#                 center_image = center_image[60:140,:,:]
                # read in steering angle
                center_angle = float(batch_sample[3])  #read the steering angle
        
                # apply a steering correction for the left and right images, in a way to generate "new" samples
                correction = 0.2
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                
                # add images and angles to batch set
                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, left_angle, right_angle])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320  # Full image format

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(row,col,ch)))
# Crop incoming data (training, validation, and autonomous so that everything is consistent)
model.add(Cropping2D(cropping=((60,20), (0,0)))) 
# Begin CNN (similar to NVIDIA architecture)
# Convolution layer 1
model.add(Conv2D(24,(5,5),strides=(2,2),activation='relu'))  #add padding='valid' ? subsample=(2,2)
# model.add(MaxPooling2D((2,2))) #max pooling to simulate stride of 2 in convolution layer
model.add(Conv2D(36,(5,5),strides=(2,2),activation='relu'))
# model.add(MaxPooling2D((2,2)))
model.add(Conv2D(48,(5,5),strides=(2,2),activation='relu'))
# model.add(MaxPooling2D((2,2),padding='vald'))
model.add(Dropout(0.5))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(100))
# model.add(Activation('relu'))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# save_path = 'model{epoch:02d}-{val_loss:.2f}.h5'
# checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)
# stopper = EarlyStopping(monitor='val_acc', min_delta=0.0003, patience=5)
## OR
batch_size = 32
epochs = 5 #***

# model.compile('adam', 'categorical_crossentropy', ['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/batch_size,validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size, epochs=5, verbose = 1) # should steps be divided by batch size?

model.save('modeld.h5')