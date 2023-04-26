# tf tools
import tensorflow as tf
# image processsing
from tensorflow.keras.preprocessing.image import (ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (VGG16)
# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model
# optimizers
from tensorflow.keras.optimizers import SGD
# scikit-learn
from sklearn.metrics import classification_report
# for plotting
import numpy as np
# path tools
import os
import pandas as pd
import sys
sys.path.append(".")
import utils.plot_function as pf

# defining function for loading data and generating
def load_data():
    # loading json metadata
    test_df = pd.read_json(os.path.join("images", "metadata", "test_data.json"), lines=True)
    train_df = pd.read_json(os.path.join("images", "metadata", "train_data.json"), lines=True)
    val_df = pd.read_json(os.path.join("images", "metadata", "val_data.json"), lines=True)
    # data generating settings
    train_datagen = ImageDataGenerator(horizontal_flip=True,
                                        rotation_range=20,
                                        rescale=1/255
    )
    test_datagen = ImageDataGenerator(
                                    rescale=1./255.
    )
    # setting image directory
    image_directory = os.path.join(".")
    # settings for sizes
    batch_size = 32
    target_size = (224, 224)
    # generating images
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
    # load model without classifier layers
    model = VGG16(include_top=False, # this removes the final classification network
                pooling='avg', # put an average pooling layer in the top instead
                input_shape=(224, 224, 3)) # changing input shape to the predefined shape of the data
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1)
    class1 = Dense(256, 
                activation='relu')(bn)
    class2 = Dense(128, 
                activation='relu')(class1)
    output = Dense(15, 
                activation='softmax')(class2)
    # define new model
    model = Model(inputs=model.inputs, 
                outputs=output)
    # compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)
    # add to model
    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    # fits the model on batches with real-time data augmentation:
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
    # predictions
    predictions = m.predict(test_img)
    predictions = np.argmax(predictions, axis=1)
    # mapping label
    labels = (train_img.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predictions]
    # classification report
    y_test = list(test.class_label)
    report = classification_report(y_test, predictions)
    return report

# defining main function
def main():
    # load and generating
    test_df, test_images, train_images, val_images = load_data()
    model, history = model_fit(train_images, val_images)
    report = output(history, model, test_images, train_images, test_df)
    # writing and saving classification report
    report_path = os.path.join("out", "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

if __name__=="__main__":
    main() 
