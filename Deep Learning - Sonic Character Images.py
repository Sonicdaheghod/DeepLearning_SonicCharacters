#!/usr/bin/env python
# coding: utf-8

# In[1]:


# #installing appropriate libraries

# pip install opencv-python
# pip install tensorflow


# In[2]:


# Right now, I am making a folder for Sonic and Tails, I will train the program to differentiate between the two characters.

#Reference video: https://youtu.be/uqomO_BZ44g?t=858


# In[3]:


#Import libraries
##organize from long to short for presentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os


# In[4]:


#import and show one of images for our program

myImage = image.load_img(r"D:\Coding Python\Machine Learning\Deep Learn\train_data\Sonic\sonic_1.jpeg")
plt.imshow(myImage)


# In[5]:


#the issues I encountered were nto having the cv library and not defining image for tensorflow.keras.preprocessing
# to resolve this i used pip install cv as well as checked and fixed the grammar in my code


# In[6]:


#output 3D shape of our image as array

cv2.imread(r"D:\Coding Python\Machine Learning\Deep Learn\train_data\Sonic\sonic_1.jpeg").shape

#(height,width,rgb)


# In[7]:


#generate the validation and training data set using ImageDataGenerator library

#we use rescale so the data points are generalized and easier to compare w/ each other

train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)


# In[8]:


#get dataset from our directory
trainDataset = train.flow_from_directory(r"D:\Coding Python\Machine Learning\Deep Learn\train_data",
                                        target_size = (200,200),
                                        batch_size = 2,
                                        class_mode = "binary")

#since I don't have that many pictures in the training file, the batch size for the machine to train with must be small
#class_mode here is binary because the only categories is either Sonic or Tails

#for validation dataset
validationDataset = train.flow_from_directory(r"D:\Coding Python\Machine Learning\Deep Learn\validation",
                                        target_size = (200,200),
                                        batch_size = 2,
                                        class_mode = "binary")


# In[9]:


#see the different classes the images are seperated into. In this case, either Tails or Sonic

trainDataset.class_indices


# In[10]:


trainDataset.classes


# In[11]:


#an error I encountered here is that the code does not process jfif files. To fix this, I changed the files to be jpeg


# In[12]:


#sequential model is for the model to bring information in order from input to output
#Conv2D is a specific kernel that uses inputs to produces tensor of outputs based on detecting image's properties (edge detect, sharpen, blur, etc)
myModel = tf.keras.models.Sequential([tf.keras.layers.Conv2D(12,(2,2), activation = "relu", input_shape = (200,200,3)),
                                                            tf.keras.layers.MaxPool2D(2,2),
                                                            
                                                            #
                                                            tf.keras.layers.Conv2D(24,(2,2), activation = "relu"),
                                                            tf.keras.layers.MaxPool2D(2,2),
                                                            
                                                            # gradual increase number of layers
                                                            tf.keras.layers.Conv2D(48,(2,2), activation = "relu"),
                                                            tf.keras.layers.MaxPool2D(2,2),
                                                                                  
                                                            ##flatten - makes out multidimension input to 1D 
                                                            tf.keras.layers.Flatten(),
                                                            ##Dense - all neurons in neural netwoek will process input of other neurons in previous layer
                                                            tf.keras.layers.Dense(512, activation= "relu"),
                                                            ##
                                                            tf.keras.layers.Dense(1, activation= "sigmoid")                     
                                                            ])
#the program learned from the training data set which pictures belong to 'Sonic' and which ones belong to "Tails"


# In[13]:


#preparing model for training with training dataset
myModel.compile(loss = "binary_crossentropy",
               optimizer = tf.keras.optimizers.RMSprop(learning_rate = .001),
               metrics = ["accuracy"])


# In[14]:


model_fit = myModel.fit(trainDataset,
                     steps_per_epoch = 3,
                     epochs = 20,
                     validation_data = validationDataset)
#epoch  = # iterations

##lower loss = better
##higher accuracy = better

###data analysis
###as go down iteration, loss begins to slightly increase and accuracy decreases after increasing

##i notice that more epochs, accuracy begins to reach 100%


# In[31]:


#determining quality of method
#for this model, I will create a function that will explicitly label what images are either Sonic or Tails

dir_path = r"D:\Coding Python\Machine Learning\Deep Learn\test_data"

#print out images one by one
for x in os.listdir(dir_path):
    theImg = image.load_img(dir_path+"//"+ x, target_size = (200,200,3))
    plt.imshow(theImg)
    plt.show()
#have model determine which pictures go where

    #convert pictures to array
    convertX = image.img_to_array(theImg)
    convertX = np.expand_dims(convertX, axis = 0)
    myImages = np.vstack([convertX])
    
    #have model explicitly print which pics are either sonic/tails
    predictX = myModel.predict(myImages)
    if predictX == 0:
        print("This is a picture of Sonic the Hedgehog.")
    else:
        print("This is a picture of Tails the Fox.")

    
   


# In[29]:


#the model incorrectly identified 3 images. The images were of SOnic the Hedgehog but the model identified them as Tails the Fox

#to fix this, I need more training data in the Sonic the Hedgehog file for the model to train with.


# In[ ]:




