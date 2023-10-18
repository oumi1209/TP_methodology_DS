import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.models
from tensorflow.keras import layers
import keras
import tensorflow as tf

num_normal_images = len(train_normal)
num_pneumonia_images = len(train_PNEUMONIA)

# Création des labels
labels_normal = np.zeros(num_normal_images, dtype=int)  
labels_pneumonia = np.ones(num_pneumonia_images, dtype=int)  

# concatenation 
labeled_images = np.concatenate((train_normal, train_PNEUMONIA), axis=0)
labels = np.concatenate((labels_normal, labels_pneumonia), axis=0)

from sklearn.model_selection import train_test_split

# Spliting the data into 80% traing  and  20% validation
labeled_images_train, labeled_images_val, labels_train, labels_val = train_test_split(
    labeled_images, labels, test_size=0.2, random_state=42)
	
# Define the input layer
input_img = keras.Input(shape=(224, 224,3))

# Convolutional layers
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(2, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)

# Flatten et fully connected layers
x = layers.Flatten()(x)
x = layers.Dense(units=128, activation='relu')(x)
x = layers.Dropout(0.2)(x)

# Output layer 
output = layers.Dense(units=1, activation='sigmoid')(x)


model = tf.keras.Model(inputs=input_img, outputs=output)

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.summary()
history=model.fit(labeled_images_train,labels_train,validation_data=(labeled_images_val, labels_val),epochs=17,batch_size=28)