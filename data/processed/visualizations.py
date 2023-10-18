import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot  as plt
import os
import cv2  

path_train_NORMAL=glob.glob("C:/Users/oumi_/Desktop/M2_SID/Methodology_DS/Data/chest_xray/train/NORMAL/*.jpeg")
path_train_PNEUMONIA=glob.glob("C:/Users/oumi_/Desktop/M2_SID/Methodology_DS/Data/chest_xray/train/PNEUMONIA/*.jpeg")


def vizualize_images(data):
    
    plt.figure(figsize= (20,10))

    for i in range(6):
        ax = plt.subplot(1, 6, i+1 )
        plt.imshow(data[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


import matplotlib.pyplot as plt

# Assuming you have two classes: 'normal' and 'pneumonia'
class_labels = ['Normal', 'Pneumonia']

# Number of samples for each class
class_counts = [len(train_normal), len(train_PNEUMONIA)]

# Create a bar plot
plt.bar(class_labels, class_counts)
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.title('Class Distribution of Data')
plt.show()


def plot_distribation_width_height(widths , heights ):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=20, color='blue', alpha=0.7)
    plt.xlabel('Image Width')
    plt.ylabel('Frequency')
    plt.title('Distribution of Image Widths')

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=20, color='green', alpha=0.7)
    plt.xlabel('Image Height')
    plt.ylabel('Frequency')
    plt.title('Distribution of Image Heights')

    plt.tight_layout()
    plt.show()