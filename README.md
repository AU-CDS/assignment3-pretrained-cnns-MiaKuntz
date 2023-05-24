# Assignment 3 - Using pretrained CNNs for image classification
This assignment focuses on classifying the Indo Fashion dataset via pretrained CNN’s. The objective is to create a Python script for the classification model, which should both be readable and reproducible, along with saving both the classification report and the training and validation history plots in the ```out``` and ```models``` folder. 

## Tasks
The tasks for this assignment are to:
-	Use a pretrained CNN to classify the data.
-	Save both training and validation history plots.
-	Save the classification as a text report.

## Repository content
The GitHub repository contains four folders, namely the ```src``` folder, which contains the Python script for using the pretrained CNN when classifying on the data, the ```out``` folder, which contains the classification report, the ```models``` folder, which contains the history plots, and the ```utils``` folder, which contains helper functions for plotting the loss and accuracy of the model history. Additionally, the repository has a ```ReadMe.md``` file, as well ```setup.sh``` and ```requirements.txt``` files.

### Repository structure
| Column | Description|
|--------|:-----------|
| ```models``` | Folder containing the history plots |
| ```out``` | Folder containing the classification report |
| ```src``` | Folder containing python script for classifying |
| ```utils``` | Folder containing utility functions for plotting loss and accuracy curves provided by the course instructor |

## Data
The data used in this assignment is the Indo Fashion dataset. This dataset contains 106,000 images spread across 15 categories pertaining to clothing items in Indian fashion.
When downloaded, the main ```archive``` repository contains another folder called ```images```, in which three subfolders can be found: “train”, “test”, and “val”. Additionally does the ```archive``` folder contain metadata in the way of three JSON files. To download the data, please following this link:

https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset

To access and prepare the data for use in the script, please; Create a user on the website, download the data, and import the data into the repository. 

## Methods
The following is a description of parts of my code where additional explanation of my decisions on arguments and functions may be needed than what is otherwise provided in the code. 

To be able to classify and plot the accuracy of the models’ predictions of the type of clothing mentioned in the dataset, I first read in the data and generate images. Please note that the code makes a few changes in the structure of the ```archive``` folder, such as creating a new metadata subfolder in it, as well as moving the JSON files to this folder, and that the script was run on a sample of the dataset, and it is therefore recommended to do the same when running, as the data is quite extensive when included as a whole. This will of course affect how the model is training, along with the results of both the plots and the classification report but will still be able to show how the code is running. The code provided in the repository is without the sample.

I then load the pretrained CNN model VGG16, add new layers, compile, and fit the model on batches of image data. The model history created when fitting, along with use of the plotting function provided for this course, is then used to create two plots: Firstly, the loss curve, and then the accuracy curve. I am also able to use it to create a classification report, which I save for later inspection of results. 

## Usage
### Prerequisites and packages
To be able to reproduce and run this code, make sure to have Bash and Python3 installed on whichever device it will be run on. Please be aware that the published script was made and run on a MacBook Pro from 2017 with the MacOS Ventura package, and that all Python code was run successfully on version 3.11.1.

The repository will need to be cloned to your device. Before running the code, please make sure that your Bash terminal is running from the repository; Afterward, please run the following from the command line to install and update necessary packages:

    bash setup.sh

### Running the script
My system requires me to type “python3” in the beginning of my commands, and the following is therefor based on this. To run the script from the command line please be aware of your specific system, and whether it is necessary to type “python3”, “python”, or something else in front of the commands. Now run:

    python3 src/clf.py

This will activate the script. When running, it will go through each of the functions in the order written in my main function. That is:
-	Processing and generating the data for each of the three ways the data has been split. The function will find the data in the ```image``` folder, where the steps on where and how to download it, yourself is described further up in the ReadMe. 
-	Generating plots for both loss and accuracy and then saving these into a single image to the ```models``` folder.
-	Creating a classification report and then saving this to the ```out``` folder.

## Results
As the code was run on a smaller sample of the data, the output in the way of the plots and classification report is somewhat lacking in their results, due to less data to train on compared to the output should the code have run on the entirety of the data.

Please note that the model in the ```models``` folder have been kept there to show a valid, although poor, example of the output from the script. This shows in the way of the curves, where the large distance between the training and validation curves indicates that the amount of data is unrepresentative. Is also shows in the classification report in the out folder, where the model’s ability to predict the labels varies, e.g., an f1-score of 0 on “dhoti_pants” and an f1-score of 1 on “blouse”. 

