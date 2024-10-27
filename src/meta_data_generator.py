####################################
# Author: Abhishek Srivastava
# Description : This file is used to call model that returns the necessary metadata.
####################################

import os
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import pickle 

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from transformers import ViTImageProcessor, ViTForImageClassification
import webcolors

class LocalMatrix:
    def __init__(self, all_images, local_matrix):
        self.all_images = all_images
        self.local_matrix = local_matrix

class Matrix:
    def __init__(self, color_to_index,cosine_similarity_exp):
        self.color_to_index = color_to_index
        self.cosine_similarity_exp = cosine_similarity_exp

class MaskNClassifier:
  def __init__(self, sam_checkpoint, model_type, device, CLASSIFICATION_MODEL_NAME):


    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    self.mask_generator = SamAutomaticMaskGenerator(sam)

    self.processor = ViTImageProcessor.from_pretrained(CLASSIFICATION_MODEL_NAME)
    self.classifier = ViTForImageClassification.from_pretrained(CLASSIFICATION_MODEL_NAME)
    self.top_category = pd.read_csv("src/category_to_top_category.csv").set_index("Name")

  def get_masks(self, image):
    masks = self.mask_generator.generate(image)
    return masks

  def classify(self, image: np.ndarray) -> tuple:
    inputs = self.processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = self.classifier(**inputs).logits

    logits = torch.softmax(logits, -1)
    confidence, idx = logits.max(-1)
    confidence, idx = confidence.item(), idx.item()
    label = self.classifier.config.id2label[idx]


    return label, confidence

  def _average_color(self, segmented_image):
    pixels = segmented_image.reshape(-1, 3).astype(float)

    non_black = pixels[np.any(pixels != [0, 0, 0], axis=1)]

    if len(non_black) > 0:
        return np.mean(non_black, axis=0).astype(int)
    else:
        return None



  def closest_colour(self, requested_colour):
      min_colours = {}
      for name in webcolors.names("css3"):
          r_c, g_c, b_c = webcolors.name_to_rgb(name)
          rd = (r_c - requested_colour[0]) ** 2
          gd = (g_c - requested_colour[1]) ** 2
          bd = (b_c - requested_colour[2]) ** 2
          min_colours[(rd + gd + bd)] = name
      return min_colours[min(min_colours.keys())]


  def _nearest_color(self, segmented_image, k=3):
    pixels = segmented_image.reshape(-1, 3)

    non_black = pixels[np.any(pixels != [0, 0, 0], axis=1)]

    if len(non_black) > 0:
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(non_black)

        # Get the colors
        colors = kmeans.cluster_centers_

        # Sort colors by their frequency
        labels = kmeans.labels_
        counts = np.bincount(labels)
        sorted_indices = np.argsort(counts)[::-1]

        return colors[sorted_indices].astype(int)[0]
    else:
        return None

  def get_dominant_color(self, segmented_image, type):
    if type == 'average':
      color = self._average_color(segmented_image)
    elif type == 'nearest':
      color = self._nearest_color(segmented_image)
    else:
      raise NotImplementedError

    if color is None:
      raise ValueError

    color_name = self.closest_colour(color)
    return color_name

  def get_metadata(self, image: np.ndarray, find_color='average'):
    masks = self.get_masks(image)

    metadata = {
        'bbox' : [],
        'label' : [],
        'condifence' : [],
        'color' : [],
        'top_category': [],
    }

    for mask in masks:
      x, y, w, h = mask['bbox']
      cropped_image = image[y:y+h, x:x+w].copy()
      if cropped_image.shape[0] < 2 or cropped_image.shape[1] < 2:
        continue
      cropped_image[~mask['segmentation'][y:y+h, x:x+w]] = 0.0
      label, confidence = self.classify(cropped_image)
      color = self.get_dominant_color(cropped_image, find_color)
      tc = self.top_category.loc[label].Category

      metadata['bbox'].append(mask['bbox'] )
      metadata['label'].append(label)
      metadata['condifence'].append(confidence)
      metadata['color'].append(color)
      metadata['top_category'].append(tc)


    return metadata

def get_image_from_path(image_path: str):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    return original_image

def get_info_for_single_image(image_path, clf):
    original_image = get_image_from_path(image_path)

    md = clf.get_metadata(original_image, find_color='average')
    md_df = pd.DataFrame(md)
    idx = md_df.groupby("top_category")["condifence"].transform(max) == md_df['condifence']
    md_df = md_df[idx]

    # Assumption - Single image will have single cloth
    md_df = md_df.query("top_category == 'Top' or top_category == 'Bottom'")
    md_dict = md_df.sort_values(by="condifence", ascending=False).head(1).to_dict('records')[0]

    return md_dict

def get_info_for_all_images(temp_dir: str):
    clf = get_model()
    meta_dict = {}
    onlyfiles = [os.path.join(temp_dir,f) for f in listdir(temp_dir) if isfile(join(temp_dir, f))]
    for f in onlyfiles:
        meta_dict[f] = get_info_for_single_image(f,clf)
    return meta_dict


def get_model():
    CLASSIFICATION_MODEL_NAME = 'jolual2747/vit-clothes-classification'
    sam_checkpoint = "src/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    clf = MaskNClassifier(sam_checkpoint, model_type, device, CLASSIFICATION_MODEL_NAME)

    return clf

def get_local_matrix(temp_dir: str) -> LocalMatrix:
    # Get meta dict for all images
    meta_dict = get_info_for_all_images(temp_dir)

    # Build K X K matrix

    #1. Load global matrix
    with open("src/global_similarity_file", 'rb') as filehandler:
        matrix = pickle.load(filehandler)
    
    all_images = list(meta_dict.keys())

    local_matrix = np.zeros((len(all_images), len(all_images)))
    local_matrix[:,:] = -np.inf

    for i, row_image in enumerate(all_images):
        for j, col_image in enumerate(all_images):

            if j < i:
                continue

            row_dict = meta_dict[row_image]
            col_dict = meta_dict[col_image]

            if i == j and row_dict['label'] != 'DRESSES':
                continue

            # if they have same type then set it to 0 and it's not same dress
            if i != j and row_dict['top_category'] == col_dict['top_category']:
                continue
            else:
                local_matrix[i,j] = matrix.cosine_similarity_exp[matrix.color_to_index[row_dict['color']], matrix.color_to_index[col_dict['color']]]
                # Dresses get 10% extra push
                if row_dict['label'] == 'DRESSES':
                    local_matrix[i,j] *= 1.1
    
    local_mat = LocalMatrix(all_images, local_matrix)

    return local_mat


if __name__ == "__main__":
    get_local_matrix("/home/paperspace/src/styler/src/tmp_dir")