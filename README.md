# Machine Learning- Deep Learning - Categorizing Characters
by Megan Tran

## Table of Contents
* [Purpose of Program](#Purpose-of-program)
* [Technologies](#technologies)
* [Setup](#setup)
* [How to Use the Program](#How-to-Use-the-Program)

## Purpose of Program

I created this program to practice creating a neural network and training it with imported photos from two categories: Sonic the Hedgehog and Tails the Fox. The goal of this was to create a program that can seperate the pictures into the two respective categories.

A challenge I faced was that the terminal did not accept ".jfif" files. To fix this, I converted the files to ".jpeg". Another challenge I faced with my model was that it incorrectly identified three images. The original pictures were of Sonic the Hedgehog (blue character), but the model identified them as Tails the Fox (yellow character). To fix this, I would have to add more pictures of "Sonic the Hedgehog" in the training file for the model to train with to improve its accuracy in identifying pictures of Sonic the Hedgehog.

In the future, I hope to incorporate and improve on this model and apply it to real life situations that incorporate categorizing certain variables into seperate groups for analysis. For example, I want to create a model similar to this in categorizing bioactive molecules and non-bioactive molecules for my research in Natural Products Chemistry.

## Technologies
Languages/ Technologies used:

*Jupyter Notebook
*Python3

## Setup

To run this project, install it locally using pip install:

`pip install tensorflow`

`pip install opencv-python`

`pip install matplotlib`

`pip install numpy`

## Using the program

1. Create a training, test, and validation file in your directory for your model to work with.

* create two files for the categories you want to include for the project. Here, I made two folders called "Sonic" and "Tails"

* In the test file, add pictures of your choice so the model can later organize it into the groups you want them organized into. 

2. Import those files to the code using their respective directories using these lines:

`myImage = image.load_img(r"D:\Coding Python\Machine Learning\Deep Learn\train_data\Sonic\sonic_1.jpeg")`

`cv2.imread(r"D:\Coding Python\Machine Learning\Deep Learn\train_data\Sonic\sonic_1.jpeg").shape`

`#get dataset from our directory
trainDataset = train.flow_from_directory(r"D:\Coding Python\Machine Learning\Deep Learn\train_data",
                                        target_size = (200,200),
                                        batch_size = 2,
                                        class_mode = "binary")`

`validationDataset = train.flow_from_directory(r"D:\Coding Python\Machine Learning\Deep Learn\validation",
                                        target_size = (200,200),
                                        batch_size = 2,
                                        class_mode = "binary")`
                                        
`dir_path = r"D:\Coding Python\Machine Learning\Deep Learn\test_data"`

3. Result should then output the model's loss/ accuracy in identifying the pictures from your custom data set. At the end when determining the quality of the model's method, it should look like this-

![sonicML](https://github.com/Sonicdaheghod/DeepLearning_SonicCharacters/assets/68253811/2a9d7548-65ea-4caa-8f41-1c84011416d5)


### Credits
This project was inspired by When Maths Meets Coding's tutorial
> [https://youtu.be/Hr06nSA-qww](https://youtu.be/uqomO_BZ44g)
