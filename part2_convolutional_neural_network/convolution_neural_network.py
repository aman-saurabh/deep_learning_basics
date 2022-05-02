#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 09:23:01 2022
@author: aman
"""
# Part1 - Importing relevent libraries
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# ImageDataGenerator is used to transform the data
from tensorflow.keras.models import Sequential
# Sequential class is required to initialize neural network
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
# Conv2D class is required to add convolution layer in the CNN
# MaxPool2D class is used to add max pooling layer in the CNN
# Flatten class is used to apply flattening in the final pooled layer
# Dense class is required to create fully connected layers in CNN(also to 
# create hidden layers in ANN)

# Part2 - Data preprocessing
# Preprocessing the training set
"""
In this step we will make some geometrical transformations in the image to 
avoid overfitting like shifting some of the pixels of the images, rotate the 
images, some horizontal flips, some zoom-in and zoom-out. So we are going to 
apply a series of transformation so as to modify the image to make them 
augmented(augmented meaning - to increase the amount, value, size, etc. of 
something). Infact the technical term for what we are going to do now is 
"Image augmentation". The purpose of doing image augmentation is that our CNN 
doesn't over-learn or over-trained on the existing image. Actually by applying 
these transformations we get new images and this way we protect our CNN from 
being over-trained on the existing images. We will use "ImageDataGenerator" 
class of keras library for the purpose. This class has different arguments 
which are used for different transformations. But we will use only few of them 
like zoom-range(used for zooming-in and zooming-out), horizontal-flip(used for 
flipping the image horizontally) and shear_range(used for shear transformation 
- details given below). Apart from them we will also use "rescale" argument 
which is used for feature scaling. Actually images are represented in pixels 
and pixels value ranges between 0-255 but as we have learnt in machine learning 
section that we should transform data in range 0-1 by applying feature-scaling.
So here we will divide the pixel value by 255 to get the values in range 0-1. 
-------------------------------------------------------------------------------
Shear transformation :-
Shear transformation slants(to lean in position or to slope) the shape of the 
image. This is different from rotation in the sense that in shear 
transformation, we fix one axis and stretch the image at a certain angle known 
as the shear angle. This creates a sort of ‘stretch’ in the image, which is not 
seen in rotation. shear_range specifies the angle of the slant in degrees.
-------------------------------------------------------------------------------
"""
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
"""
Here 'train_datagen' is nothing but an instance of ImageDataGenerator class.
which we will use to transform our images below.
"""
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
"""
Here 'flow_from_directory' is a method of class ImageDataGenerator. In this 
first argument is the traning set location, second argument is target_size 
of augmented image(in pixels). third argument is batch size i.e how how many 
images we want to process in a batch and fourth and last argument is 
class_mode which we set as 'binary' as output will have only two options(cat 
or dog). If there were multiple options we would have used 'categorical'. 
Actually "categorical" will be 2D one-hot encoded labels, - "binary" will be 
1D binary labels, "sparse" will be 1D integer labels.  
"""

#Preprocessing the test set
"""
In test set also we won't transform the image as we want to test our model on 
original provided test images itself. However we will apply feature scaling as 
we want image's pixels value in range of 0-1 and not in the range of 0-255.   
"""
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part3 - Building the CNN
#Initializing the CNN
cnn = Sequential()

# Step 1 - Convolution
cnn.add(Conv2D(filters=32, 
               kernel_size=3, 
               activation='relu', 
               input_shape=[64, 64, 3]))
"""
Here first parameter 'filters' specifies the number of required filter(i.e 
feature detectors) in this convolution layer . Here we can choose any number of 
filters but we should follow some predefined CNN architecture. Here we are 
following some classic architecture so we have choosen number of filters as 32.
second parameter is 'kernel_size' which is used to set the size of 
feature-detector, here we have choosen 3, which will create a feature detector 
of size 3*3 (as feature-detectors are normally of square shape). Third parameter 
is 'activation' which is used to set the activation function. Here we have 
choosen 'relu' activation function which stands for "Rectifier linear unit". 
and fourth and last argument is 'input_shape' which is used to specify the shape 
of the input image.Actually it is used to reshape the input_image to 4D and it 
accepts the data in format [batch_size, img_height, img_width, number_of_channels] 
Since we have transformed the input image size to (64*64) pixels so our 
img_height and img_width is 64 and since our image is RGB-image(i.e color image) 
so here 'number_of_channels' will be 3. For black and white image we should use 
1. And we can omit 'batch_size' so we have not used that.    
# Note :- To know more about CNN architecture - Read about it on google.     
"""

# Step2 - Max Pooling
cnn.add(MaxPool2D(pool_size=2, strides=2))

# Step3 - Adding a second convolutional layer and max pooling layer
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
"""
Since it is a second convolution layer and won't interact with input directly 
so we don't need 'input_shape' parameter here.
"""
cnn.add(MaxPool2D(pool_size=2, strides=2))

# Step4 - Flattening
cnn.add(Flatten())

# Step5 - Fully connected layer
cnn.add(Dense(128, activation='relu'))
"""
Since CNN is quiet complex in comparison to simple ANN so we choosed units=128 
which is nothing but number of nodes you want in the current fully connected 
layer(which is nothing but hidden layer of ANN). And since it is not the final 
layer so we choosed activation function as 'relu'.
For details of Dense class check its application in ANN program.
"""

# Step6 - Output layer
cnn.add(Dense(1, activation='sigmoid'))
"""
Since it is a binary classification problem i.e output have only two possible 
values. So we can represent it using 1 output only. So we choosed units=1 and 
due to same reason only we choosed activation='sigmoid'. If output had multiple 
possible values, we would have choosen 'softmax' inplace of 'sigmoid'. 
"""

# Part4 - Training the CNN
# Compiling ANN (i.e applying Stochastic Gradient Descent to the ANN) 
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# For details check this part of ANN program.

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)
"""
Here we are performing training of the model and evaluation of the model, both 
simultaneously. First argument 'x' is used to specify the training data and 
'validation_data' is used to specify the dataset on which model will be 
evaluated. For third parameter you can choose 'epochs' value to any number but 
you should choose the value for which result is optimal. We have tested and 
found 25 as the suitable choice.  
"""

# Part5 - Making a single prediction 
import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', 
                            target_size = (64, 64))
test_image = image.img_to_array(test_image)
"""
In predict method we need input data in the form of numpy array.We can 
achieve that using 'img_to_array' method of keras image module.
"""
test_image = np.expand_dims(test_image, axis = 0)
"""
Our CNN is trained to accept images in batch. So we need an extra dimension 
in the above created numpy array to represent the batch.We can achieve that 
using numpy's expand_dims method as above.
So now we have input in the right format expected by the predict method.
"""
result = cnn.predict(test_image/255.0)   
""" 
Here we are dividing the test image by 255 to normalize it i.e to get the 
pixels value in range 0-1 instead of 0-255. 
It will produce result but in range of 0 - 1. Value near to 0 will represent 1 
category and value near to 1 will represent another. But now the question is 
how we will know whether 0 corresponds to Cat or 0 corresponds to Dog. We can 
get this information by using 'class_indices' attribute of our training set 
object i.e if we run "print(training_set.class_indices)" then it will print 
whether 0 corresponds to Cat or Dog and same for 1.
"""
print(training_set.class_indices)
"""
Suppose above code printed 1 corresponds to Dog and 0 corresponds to Cat. So
now we can print whether the image represents Dog or cat as follows:  
"""
if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
"""
As result is also in the batch format so first [0] represents the one and only 
batch and since the result has only one value so it must be at index 0. 
"""