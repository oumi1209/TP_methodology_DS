import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot  as plt
import os
import cv2

path_train_NORMAL=glob.glob("C:/Users/oumi_/Desktop/M2_SID/Methodology_DS/Data/chest_xray/train/NORMAL/*.jpeg")
path_train_PNEUMONIA=glob.glob("C:/Users/oumi_/Desktop/M2_SID/Methodology_DS/Data/chest_xray/train/PNEUMONIA/*.jpeg")


def read_images(path):
    
    train_classe = []

    target_size = (224, 224)

    for image in path:
        img = cv2.imread(image)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        # Resize the image to the target size
        resized_img = cv2.resize(rgb, target_size)
    
        train_classe.append(resized_img)

     # Convert the list of resized images to a NumPy array
    train_classe = np.array(train_classe)
    train_classe_normalize=train_classe.astype('float32')/255
    return(train_classe_normalize)

def vizualize_images(data):
    
    plt.figure(figsize= (20,10))

    for i in range(6):
        ax = plt.subplot(1, 6, i+1 )
        plt.imshow(data[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
		
train_normal=read_images(path_train_NORMAL)
vizualize_images(train_normal)

train_PNEUMONIA=read_images(path_train_PNEUMONIA)
vizualize_images(train_PNEUMONIA)
    