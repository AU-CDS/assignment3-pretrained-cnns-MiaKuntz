# Author: Mia Kuntz
# Date hand-in: 24/5 - 2023

# Description: This script is used for training a classifier on the data.
# The script loads the data, fits the model and generates a classification report.

# tensorflow
import tensorflow as tf 
# image processsing tools
from tensorflow.keras.preprocessing.image import (ImageDataGenerator)
# VGG16 model 
from tensorflow.keras.applications.vgg16 import (VGG16)
# layers for model
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     BatchNormalization)
# generic model object 
from tensorflow.keras.models import Model
# optimizers 
from tensorflow.keras.optimizers import SGD
# scikit-learn for classification report
from sklearn.metrics import classification_report
# plotting history
import numpy as np
# operating system
import os
# pandas for dataframes
import pandas as pd
# adding path to sys
import sys
sys.path.append(".")
# plotting function
import utils.plot_function as pf

# defining function for loading data and generating images
def load_data():
    # loading json metadata files
    test_df = pd.read_json(os.path.join("images", "metadata", "test_data.json"), lines=True)
    train_df = pd.read_json(os.path.join("images", "metadata", "train_data.json"), lines=True)
    val_df = pd.read_json(os.path.join("images", "metadata", "val_data.json"), lines=True)
    # generating settings for images
    train_datagen = ImageDataGenerator(horizontal_flip=True,
                                        rotation_range=20,
                                        rescale=1/255
    )
    test_datagen = ImageDataGenerator(
                                    rescale=1./255.
    )
    # setting image directory 
    image_directory = os.path.join(".")
    # settings for sizes and batch
    batch_size = 32
    target_size = (224, 224)
    # generating images from dataframes
    test_images = test_datagen.flow_from_dataframe(
        dataframe = test_df,
        directory = image_directory,
        x_col = "image_path",
        y_col = "class_label",
        target_size = target_size,
        color_mode = "rgb",
        class_mode = "categorical",
        batch_size = batch_size,
        shuffle = False,
    )
    train_images = train_datagen.flow_from_dataframe(
        dataframe = train_df,
        directory = image_directory,
        x_col = "image_path",
        y_col = "class_label",
        target_size = target_size,
        color_mode = "rgb",
        class_mode = "categorical",
        batch_size = batch_size,
        shuffle = True,
        seed = 42,
        subset = "training"
    )
    val_images = train_datagen.flow_from_dataframe(
        dataframe = val_df,
        directory = image_directory,
        x_col = "image_path",
        y_col = "class_label",
        target_size = target_size,
        color_mode = "rgb",
        class_mode = "categorical",
        batch_size = batch_size,
        shuffle = True,
        seed = 42,
    )
    return test_df, test_images, train_images, val_images

# defining function for pretrained model and fitting
def model_fit(train_img, val_img):
    # loading model
    model = VGG16(include_top=False, # removing final classification network
                pooling='avg', # putting average pooling layer in the top instead
                input_shape=(224, 224, 3)) # changing input shape to the predefined shape of the data
    # marking loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # adding new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    # adding batch normalization
    bn = BatchNormalization()(flat1)
    # adding dense layers
    class1 = Dense(256, 
                activation='relu')(bn)
    class2 = Dense(128, 
                activation='relu')(class1)
    output = Dense(15, 
                activation='softmax')(class2)
    # defining new model with the classifier layers
    model = Model(inputs=model.inputs, 
                outputs=output)
    # compiling model
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01, # starting learning rate
        decay_steps=10000, # decay steps
        decay_rate=0.9) # decay rate
    # stochastic gradient descent
    sgd = SGD(learning_rate=lr_schedule)
    # compiling model
    model.compile(optimizer=sgd, 
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    # fitting model on batches with real-time data augmentation
    history = model.fit(
        train_img,
        steps_per_epoch = len(train_img),
        validation_data = val_img,
        validation_steps = len(val_img),
        epochs = 10)
    return model, history

# defining function for plotting and classification report
def output(h, m, test_img, train_img, test):
    pf.plot_history(h, 10, "la_history_plot")
    # predicting on test data
    predictions = m.predict(test_img)
    predictions = np.argmax(predictions, axis=1)
    # mapping label
    labels = (train_img.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    # generating predictions
    predictions = [labels[k] for k in predictions]
    # generating classification report
    y_test = list(test.class_label)
    report = classification_report(y_test, predictions)
    return report

# defining main function
def main():
    # loading data
    test_df, test_images, train_images, val_images = load_data()
    # fitting model
    model, history = model_fit(train_images, val_images)
    # generating classification report
    report = output(history, model, test_images, train_images, test_df)
    # writing and saving classification report
    report_path = os.path.join("out", "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

if __name__=="__main__":
    main() 

# Command line argument: 
# python3 src/clf.py