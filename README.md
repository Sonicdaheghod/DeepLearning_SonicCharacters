# Machine Learning - Categorizing Characters
by Megan Tran

## Table of Contents
* [Purpose of Program](#Purpose-of-program)
* [Technologies](#technologies)
* [Setup](#setup)
* [How to Use the Program](#How-to-Use-the-Program)

## Purpose of Program

I created this program to practice creating a neural network and training it with imported photos from two categories: Sonic the Hedgehog and Tails the Fox.

A challenge I faced what pulling up data for the Game Number when assigning the test and the train data. The program did not register the title of the columns I typed in the terminal. To fix that, I edited the column names in the csv file itself so that the title I typed in the terminal would match the column name in the csv file and run the program. 

In the future, I hope to incorporate more analysis techniques related to machine learning to work with data such as the data set I worked with to better analyze it. I would also use the new machine learning techniques to train the model to both interpolate and extrapolate data.

## Technologies
Languages/ Technologies used:

*Jupyter Notebook
*Python3

## Setup

1. Download the csv file of your choice from this website:

> https://people.sc.fsu.edu/~jburkardt/data/csv/csv.html

I downloaded snakes_count_1000.csv. for this project.

2. Ensure to import pandas
` import pandas as pd `

3. When opening csv file, save the file then right click on file in file explorer, click on "copy by path", and paste it into the code.

`game = pd.read_csv(r"C:\Users\Megan Tran\Desktop\Megan's USB\College\Code\Python\Machine Learning\snakes_count_1000.csv")`

### Credits
This project was inspired by Dataquest's tutorial
> https://youtu.be/Hr06nSA-qww
