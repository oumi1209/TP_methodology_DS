# Your code here for Data Augmentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen_normal = ImageDataGenerator(
    rotation_range=20,           
    width_shift_range=0.2,       
    height_shift_range=0.2,      
    shear_range=0.2,             
    zoom_range=0.2,           
    horizontal_flip=True,      
    fill_mode='nearest'        
)