import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
from mrcnn.config import Config
from mrcnn import model as modellib, utils


# Define the Mask R-CNN configuration class
class RoofConfig(Config):
    NAME = "roof"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  # Background + roof
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 100
    MAX_GT_INSTANCES = 10
    POST_NMS_ROIS_INFERENCE = 100
    POST_NMS_ROIS_TRAINING = 1000
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_NMS_THRESHOLD = 0.3
    LEARNING_RATE = 0.001
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.0,
        "rpn_bbox_loss": 1.0,
        "mrcnn_class_loss": 1.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 1.0
    }

# Define the Mask R-CNN model class
class RoofModel(object):
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        # Define the inference configuration
        inference_config = RoofConfig()
        inference_config.display()

        # Create the Mask R-CNN model in inference mode
        self.model = modellib.MaskRCNN(mode="inference", model_dir=model_path, config=inference_config)

        # Load the trained weights into the model
        self.model.load_weights(model_path, by_name=True)

    def detect_roof(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to the image
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Perform inference on the image using the Mask R-CNN model
        results = self.model.detect([image], verbose=1)

        # Get the first result from the list
        r = results[0]

        # Iterate over the detected instances and draw the masks on the original image
        for i in range(r['rois'].shape[0]):
            height = r['rois'][i][2] - r['rois'][i][0]
            if height > 2:
                mask = r['masks'][:, :, i]
                image = apply_mask(image, mask, color=[0, 255, 0])

        return image


# Define a utility function to apply the mask to the image
def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c])
    return image


# Load the Mask R
