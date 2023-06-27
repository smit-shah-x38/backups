# %%
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Commonly used modules
import numpy as np
import os
import sys

# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2
import IPython
from six.moves import urllib

print(tf.__version__)

# %%
# Code for loading Data

dataset = keras.utils.image_dataset_from_directory(
    directory="/var/trainingData/",
    labels="inferred",
    label_mode="int",
    batch_size=32,
    shuffle=True,
    seed=123,
)

# %%
test_dataset = dataset.take(500)
train_dataset = dataset.skip(1000)
train_dataset = train_dataset.take(5000)

# %%
print(len(train_dataset))
print(len(test_dataset))


# %%
# Define a custom layer that wraps the resize_with_pad function
class ResizeWithPadLayer(tf.keras.layers.Layer):
    def __init__(self, target_height, target_width):
        super(ResizeWithPadLayer, self).__init__()
        self.target_height = target_height
        self.target_width = target_width

    def call(self, inputs):
        return tf.image.resize_with_pad(inputs, self.target_height, self.target_width)


# Create an instance of the custom layer
resize_with_pad_layer = ResizeWithPadLayer(180, 180)


# %%
model = keras.Sequential()
model.add(resize_with_pad_layer)
# 32 convolution filters used each of size 3x3
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(180, 180, 1)))
# 64 convolution filters used each of size 3x3
model.add(Conv2D(64, (3, 3), activation="relu"))
# choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# randomly turn neurons on and off to improve convergence
model.add(Dropout(0.25))
# flatten since too many dimensions, we only want a classification output
model.add(Flatten())
# fully connected to get all relevant data
model.add(Dense(128, activation="relu"))
# one more dropout
model.add(Dropout(0.5))
# output a softmax to squash the matrix into output probabilities
model.add(Dense(10, activation="softmax"))

# %%
model.compile(
    optimizer=tf.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# %%
history = model.fit(train_dataset, epochs=1)

# %%
