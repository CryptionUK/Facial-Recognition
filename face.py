import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation

import numpy as np
import imutils
from sklearn import utils
from numpy import asarray
import cv2 as cv
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
#from matplotlib.pyplot import imread, imshow, subplots, show

trainImgDir = "C:/10759994/Dissertation/images/train/"
train_image = []
testImgDir = "C:/10759994/Dissertation/images/test/"
test_image = []

# Defining an array for visual purposes to display which image is which
people = ['Chris Evans', 'Chris Hemsworth', 'Mark Ruffalo', 'Robert Downey Jr', 'Scarlett Johansson']

# Defining arrays to store the labels for both training and testing
train_labels = []
test_labels = []

img_cols = 100
img_rows = 100
size = (100,100)

for i, image in tqdm(enumerate(os.listdir(trainImgDir))):

    img = cv.imread(trainImgDir + image, 3)

    # Creating the label from the file name by removing the first part of the file name (before the space)
    lbl = image.split(" ")

    # Append the label to the training labels array
    train_labels.append(int(lbl[0]))

    # Grab the height and width of the image
    height, width = img.shape[:2]

    # If the height or image is greater than 100 run through the following code
    if (height > 100) or (width > 100):

        # If the width of the image is greater than the height then set the width to be 175 and keep the aspect ratio
        if width > height:
            img = imutils.resize(img, width=100)
        else:
            img = imutils.resize(img, height=100)

        # Grab the height and width of the image
        height, width = img.shape[:2]
    
        # Create a blank background with the size of 100 x 100
        blank_image = np.zeros((100, 100, 3), np.uint8)
        # Sets the colour of the background to have an rgb of 0,0,0
        blank_image[:,:] = (0, 0, 0)

        # Make a new image with a copy of the grey background
        newImg = blank_image.copy()
        # Set the offset of the image to 0
        x_offset = y_offset = 0
        newImg[y_offset:y_offset+height, x_offset:x_offset+width] = img.copy()

        # Converts the images to a one-dimensional array for the model to do it's magic
        if K.image_data_format() == 'channels_first':
            trained = newImg.reshape(3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            trained = newImg.reshape(img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)

    else:

        # Create a blank background with the size of 100 x 100
        blank_image = np.zeros((100, 100, 3), np.uint8)
        # Sets the colour of the background to have an rgb of 0,0,0
        blank_image[:,:] = (0, 0, 0)

        # Make a new image with a copy of the grey background
        newImg = blank_image.copy()
        # Set the offset of the image to 0
        x_offset = y_offset = 0
        newImg[y_offset:y_offset+height, x_offset:x_offset+width] = img.copy()

        # Converts the images to a one-dimensional array for the model to do it's magic
        if K.image_data_format() == 'channels_first':
            trained = newImg.reshape(3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            trained = newImg.reshape(img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)

    # Append the tested image to the array
    train_image.append(asarray(trained))

train_image = np.array(train_image)

for i, image in tqdm(enumerate(os.listdir(testImgDir))):

    img = cv.imread(testImgDir + image, 3)

    # Creating the label from the file name by removing the first part of the file name (before the space)
    lbl = image.split(" ")

    # Append the label to the test labels array
    test_labels.append(int(lbl[0]))

    # Grab the height and width of the image
    height, width = img.shape[:2]

    # If the height or image is greater than 100 run through the following code
    if (height > 100) or (width > 100):

        # If the width of the image is greater than the height then set the width to be 175 and keep the aspect ratio
        if width > height:
            img = imutils.resize(img, width=100)
        else:
            img = imutils.resize(img, height=100)

        # Grab the height and width of the image
        height, width = img.shape[:2]
    
        # Create a blank background with the size of 100 x 100
        blank_image = np.zeros((100, 100, 3), np.uint8)
        # Sets the colour of the background to have an rgb of 0,0,0
        blank_image[:,:] = (0, 0, 0)

        # Make a new image with a copy of the grey background
        newImg = blank_image.copy()
        # Set the offset of the image to 0
        x_offset = y_offset = 0
        newImg[y_offset:y_offset+height, x_offset:x_offset+width] = img.copy()

        if K.image_data_format() == 'channels_first':
            tested = newImg.reshape(3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            tested = newImg.reshape(img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)

    else:

        # Create a blank background with the size of 100 x 100
        blank_image = np.zeros((100, 100, 3), np.uint8)
        # Sets the colour of the background to have an rgb of 0,0,0
        blank_image[:,:] = (0, 0, 0)

        # Make a new image with a copy of the grey background
        newImg = blank_image.copy()
        # Set the offset of the image to 0
        x_offset = y_offset = 0
        newImg[y_offset:y_offset+height, x_offset:x_offset+width] = img.copy()

        if K.image_data_format() == 'channels_first':
            tested = newImg.reshape(3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            tested = newImg.reshape(img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)

    # Append the tested image to the array
    test_image.append(asarray(tested))

# Converting the arrays to a numpy array for the model
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
train_image = np.array(train_image)
test_image = np.array(test_image)

train_image, train_labels = utils.shuffle(train_image, train_labels)
test_image, test_labels = utils.shuffle(test_image, test_labels)

def decay(epoch):
    if epoch < 11:
        return 0.001
    elif epoch >= 11 and epoch < 30:
        return 0.0001
    elif epoch >= 30 and epoch < 70:
        return 0.00001
    else:
        return 0.00001

plt.figure(figsize=(10,10))
for b in range(25):
    plt.subplot(5,5,b+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(cv.cvtColor(train_image[b], cv.COLOR_BGR2RGB))
    plt.xlabel(people[train_labels[b]])
plt.show()

# Running the model
model = models.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(256))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
callbacks = [tf.keras.callbacks.LearningRateScheduler(decay)]
history = model.fit(train_image, train_labels, epochs=100, validation_data=(test_image, test_labels))
test_loss, test_acc = model.evaluate(test_image, test_labels, verbose=2)

# Convert the test accuracy to a percentage
testaccuracy = float("{:.2f}".format(test_acc*100))

print('\nTest accuracy:', str(testaccuracy) + '%')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()