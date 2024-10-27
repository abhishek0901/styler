####################################
# Author: Abhishek Srivastava
# Description : This file will generate metadata for single image
####################################

from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import cv2
import numpy as np
import webcolors


CLOTH_MAP = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "DRESSES", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"
}

CLOTH_OF_INTEREST = [4,5,6,7]

def rgb_to_color(requested_colour):
      min_colours = {}
      for name in webcolors.names("css3"):
          r_c, g_c, b_c = webcolors.name_to_rgb(name)
          rd = (r_c - requested_colour[0]) ** 2
          gd = (g_c - requested_colour[1]) ** 2
          bd = (b_c - requested_colour[2]) ** 2
          min_colours[(rd + gd + bd)] = name
      return min_colours[min(min_colours.keys())]

def get_metadata(image_path: str) -> dict:
    """
    Description
    --------------
        1. Load model
        2. Get Predictions
        3. Get Type
        4. Get Color
        5. Fill Values
        6. Return Dict
    """

    # Load Image
    image = Image.open(image_path)
    # Load model
    processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")
    # Process
    inputs = processor(images=image, return_tensors="pt")

    # get predictions
    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    # Get type
    cloth_type_predicted_index = torch.mean(upsampled_logits[0,CLOTH_OF_INTEREST], dim=[1,2]).argmax()
    print(f"Cloth is : {CLOTH_MAP[CLOTH_OF_INTEREST[cloth_type_predicted_index]]}")
    cloth_type = CLOTH_MAP[CLOTH_OF_INTEREST[cloth_type_predicted_index]]

    # get color
    t1 = upsampled_logits[0,CLOTH_OF_INTEREST[cloth_type_predicted_index]].detach().numpy()
    t1[t1 < t1.mean()] = 0
    t1[t1 >= t1.mean()] = 1
    t2 = t1.reshape((t1.shape[0], t1.shape[1], 1))
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_copy = original_image * t2
    rgb_values = original_image_copy[np.nonzero(t1)].mean(axis=0)

    #convert them to int
    rgb_values = rgb_values.astype(int)
    print(rgb_values)
    color = rgb_to_color(rgb_values)

    # Meta data
    metadata = {
        'bbox' : [],
        'label' : 'default',
        'condifence' : 0.9,
        'color' : 'default',
        'top_category': 'default',
    }

    metadata["label"] = cloth_type
    metadata["color"] = color
    metadata["top_category"] = 'Top' if cloth_type in ['Upper-clothes', 'DRESSES'] else 'Bottom'

    return metadata







if __name__ == "__main__":
    print(get_metadata("/home/paperspace/src/styler/src/tmp_dir/green_shirt.png"))