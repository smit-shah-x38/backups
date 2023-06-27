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
# # Code for loading Data
# dataset = keras.preprocessing.image_dataset_from_directory(
#     directory="/var/trainingData/",
#     labels="inferred",
#     label_mode="int",
#     batch_size=32,
#     image_size=(180, 180),
#     shuffle=True,
#     seed=123,
# )

# %%
# Assume your directory structure is:
# main_directory/
# ...class_a/
# ......a_image_1.jpg
# ......a_image_2.jpg
# ...class_b/
# ......b_image_1.jpg
# ......b_image_2.jpg

# Create a training dataset from the main directory with 70% of data
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/var/trainingData/",
    labels="inferred",
    label_mode="int",
    image_size=(180, 180),
    batch_size=32,
    shuffle=True,
    seed=123,
    validation_split=0.003,
    subset="validation",
)

# Create a validation dataset from the main directory with 30% of data
val_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/var/trainingData/",
    labels="inferred",
    label_mode="int",
    image_size=(180, 180),
    batch_size=32,
    shuffle=True,
    seed=123,
    validation_split=0.0003,
    subset="validation",
)

# %%
# Get the class names from the train_ds
class_names = train_ds.class_names

# Print the class names
print(class_names)

# Get the number of labels
num_labels = len(class_names)

# Print the number of labels
print(num_labels)


# %%
# Define a filter function that checks if the image has a non-zero shape
def filter_zero_images(image, label):
    # Get the shape of the image
    image_shape = tf.shape(image)
    # Check if the shape is non-zero
    non_zero = tf.math.reduce_any(image_shape > 0)
    # Return True or False
    return non_zero


# Apply the filter function to your batch dataset
train_ds = train_ds.filter(filter_zero_images)

# %%


# %%
# Get the number of elements in the dataset as a tensor
num_elements = tf.data.experimental.cardinality(train_ds)

# Convert the tensor to a Python integer
num_elements = tf.get_static_value(num_elements)

# Print the number of elements
print(num_elements)


# %%
# Get the batch size
batch_size = 32

# Get the number of images by multiplying the number of elements by the batch size
num_images = num_elements * batch_size

# Print the number of images
print(num_images)


# %%
# import torchvision.transforms as transforms

# # Define a transform to convert PIL images to tensors
# transform = transforms.ToTensor()

# # Iterate through each sample in the train_dataset
# for sample in train_dataset:
#     # Get the image tensor from the sample
#     image_tensor = sample[0]

#     # Check if the image tensor is empty
#     if image_tensor.numel() == 0:
#         print("Empty image found!")

#     # If you want to convert the image tensor to a PIL image for further processing:
#     # Convert the image tensor to a PIL image
#     image_pil = transforms.ToPILImage()(image_tensor)

#     # Check if the PIL image dimensions are zero
#     width, height = image_pil.size
#     if width == 0 or height == 0:
#         print("Empty image found!")


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
# Import the necessary modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model architecture
model = keras.Sequential(
    [
        # Rescale the pixel values to the range [0, 1]
        layers.Rescaling(1.0 / 255, input_shape=(180, 180, 3)),
        # Apply a convolutional layer with 16 filters and a 3x3 kernel
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        # Apply a max pooling layer with a 2x2 window
        layers.MaxPooling2D(),
        # Apply another convolutional layer with 32 filters and a 3x3 kernel
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        # Apply another max pooling layer with a 2x2 window
        layers.MaxPooling2D(),
        # Apply a dropout layer with a rate of 0.2 to reduce overfitting
        layers.Dropout(0.2),
        # Flatten the output of the previous layer
        layers.Flatten(),
        # Apply a dense layer with 128 units and a ReLU activation
        layers.Dense(128, activation="relu"),
        # Apply another dropout layer with a rate of 0.2
        layers.Dropout(0.2),
        # Apply an output layer with the number of classes and a softmax activation
        layers.Dense(2002, activation="softmax"),
    ]
)

# Compile the model with an optimizer, a loss function, and a metric
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Print the model summary
model.summary()


# %%
# Train the model for 10 epochs
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(val_ds)

# Print the loss and accuracy
print("Loss: ", loss)
print("Accuracy: ", accuracy)


# %%
