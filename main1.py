import cv2
import numpy as np
import json
import pandas as pd
import random




# Load the annotations from a COCO JSON file
with open('export-2023-02-26T21_56_25.914Z.json', 'r') as f:
    dataset = json.load(f)


image1 = cv2.imread('1.jpg')
image2 = cv2.imread('2.jpg')
image3 = cv2.imread('3.jpg')
image4 = cv2.imread('4.jpg')
image5 = cv2.imread('5.jpg')
image6 = cv2.imread('6.jpg')

images = [image1,image2,image3,image4,image5,image6]


random.shuffle(dataset)

# Calculate the number of samples to use for training and validation
num_train_samples = int(0.7 * len(dataset))
num_val_samples = len(dataset) - num_train_samples

# Split the dataset into training and validation sets
train_set = dataset[:num_train_samples]
val_set = dataset[num_train_samples:]

# Print the number of samples in each set
print(f"Number of training samples: {len(train_set)}")
print(f"Number of validation samples: {len(val_set)}")





