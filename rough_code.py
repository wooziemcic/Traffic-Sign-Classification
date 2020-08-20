# 1. Import libraries

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import pickle
import random

# 2. Divide data into train validation and test

# 60%, 20%, 20% 
'''
training set - used for gradient calc and weight update
validation set - for cross validation to over-come over-fitting
testing set - used for testing trained network
'''

# rb - read binary mode
with open("./traffic-signs-data/train.p", mode='rb') as training_data:
    train = pickle.load(training_data)
with open("./traffic-signs-data/valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open("./traffic-signs-data/test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# 3. Perform image visualization

# for one image
i = np.random.randint(1, len(X_train))
plt.imshow(X_train[i])
y_train[i]

#creating a 5 * 5 grid
# Define the dimensions of the plot grid 
W_grid = 5
L_grid = 5

# fig, axes = plt.subplots(L_grid, W_grid)
# subplot return the figure object and axes object
# we can use the axes object to plot specific figures at various locations

fig, axes = plt.subplots(L_grid, W_grid, figsize = (10,10))

axes = axes.ravel() # flaten the 5 x 5 matrix into 25 array

n_training = len(X_train) # get the length of the training dataset

# Select a random number from 0 to n_training
# create evenly spaces variables 
for i in np.arange(0, W_grid * L_grid):
    # Select a random number
    index = np.random.randint(0,n_training)
    # read and display an image with the selected index    
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index], fontsize = 15)
    axes[i].axis('off')

plt.subplots_adjust(hspace = 0.4)

# 4. Convert images to grayscale and perform normalization

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

# convert to grayscale
X_train_gray = np.sum(X_train/3, axis = 3, keepdims = True)
X_validation_gray = np.sum(X_validation/3, axis = 3, keepdims=True)
X_test_gray = np.sum(X_test/3, axis = 3, keepdims = True)

# normalize
X_train_gray_norm = (X_train_gray-128)/128
X_validation_gray_norm = (X_validation_gray-128)/128
X_test_gray_norm = (X_test-128)/128

# visualizing the normal, grayscale and the normalized grayscale images
i = random.randint(1, len(X_train_gray))
plt.imshow(X_train_gray[i].squeeze(), cmap = 'gray')
plt.figure()
plt.imshow(X_train[i])
plt.figure()
plt.imshow(X_train_gray_norm[i].squeeze(), cmap = 'gray')

# 5. Understanding CNN
'''
image => feature detection and attraction => relu function => pooling layer/downsampling => flattening => output
'''

'''
	Dropout Technique
		Imporve the accuracy by adding dropout
		Dropout refers to dropping our units in a neural network
		Neurons develop a co-dependency amongst each other during training
		Dropout is a regularization technique for reducing overfitting in neural networks
'''

# Visualize online:
# https://www.cs.ryerson.ca/~aharley/vis/conv/flat.html

# 6. Build deep convolutional neural network model

from tensorflow.keras import datasets, layers, models

CNN = models.Sequential() # to make it sequential

# layer 1 
CNN.add(layers.Conv2D(6, (5,5), activation = 'relu', input_shape=(32,32,1)))
CNN.add(layers.AveragePooling2D())

# dropout
CNN.add(layers.Dropout(0.2))

# layer 2
CNN.add(layers.Conv2D(16, (5,5),activation = 'relu')) #input_shape needed to be defined only once
CNN.add(layers.AveragePooling2D())


# flatten
CNN.add(layers.Flatten())

# pass on to the input and hidden layers
CNN.add(layers.Dense(120, activation ='relu'))
CNN.add(layers.Dense(84, activation='relu'))

#for the output
CNN.add(layers.Dense(43, activation='softmax')) 
CNN.summary()

# 7. Compile and train the CNN model

CNN.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

history = CNN.fit(X_train_gray_norm,
                 y_train,
                 batch_size = 500,
                 epochs = 5,
                 verbose = 1,
                 validation_data = (X_validation_gray_norm, y_validation))

# 8. Assess trained CNN model performance

score = CNN.evaluate(X_test_gray_norm, y_test)
print('Test Accuracy: {}'.format(score[1]))

history.history.keys()

ccuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, loss, 'ro', label = 'Training Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Training and Validation Loss')

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'ro', label = 'Training Loss')
plt.plot(epochs, val_accuracy, 'r', label = 'Validation Loss')
plt.title('Training and Validation Loss')

predicted_classes = CNN.predict_classes(X_test_gray_norm)
y_true = y_test

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize = (25, 25))
sns.heatmap(cm, annot = True)

L = 5
W = 5

fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction = {}\n True = {}'.format(predicted_classes[i], y_true[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 1)    



