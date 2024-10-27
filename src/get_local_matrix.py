####################################
# Author: Abhishek Srivastava
# Description : This file will generate local matrix
####################################

import numpy as np
import pickle

def get_local_matrix(meta_data_list: list[dict]) -> [np.ndarray, list]:

    """
    Description
    -------------
        1. Load global matrix
        2. Fill local matrxi based on color combination
    """
    # Global matrix
    with open("src/global_similarity_file", 'rb') as filehandler:
        matrix = pickle.load(filehandler)
    
    all_images = [a["image_name"] for a in meta_data_list]
    # Local Matrix
    local_matrix = np.zeros((len(all_images), len(all_images)))
    local_matrix[:,:] = -np.inf

    for i, row_obj in enumerate(meta_data_list):
        for j, col_obj in enumerate(meta_data_list):
            if j < i:
                continue

            row_dict = meta_data_list[i]
            col_dict = meta_data_list[j]

            if i == j and row_dict['label'] != 'DRESSES':
                continue

            if i != j and row_dict['top_category'] == col_dict['top_category']:
                continue
            else:
                local_matrix[i,j] = matrix.cosine_similarity_exp[matrix.color_to_index[row_dict['color']], matrix.color_to_index[col_dict['color']]]
                # Dresses get 10% extra push
                if row_dict['label'] == 'DRESSES':
                    local_matrix[i,j] *= 1.1
    
    return local_matrix, all_images


if __name__ == "__main__":
    meta_data_list = []
    elem1 = {
        'image_name':'khaki_pant', 'bbox': [], 'label': 'Pants', 'condifence': 0.9, 'color': 'peru', 'top_category': 'Bottom'
    }
    elem2 = {
        'image_name':'blue_shirt', 'bbox': [], 'label': 'Upper-clothes', 'condifence': 0.9, 'color': 'darkslategrey', 'top_category': 'Top'
    }
    elem3 = {
        'image_name':'green_shirt', 'bbox': [], 'label': 'Upper-clothes', 'condifence': 0.9, 'color': 'mediumseagreen', 'top_category': 'Top'
    }
    elem4 = {
        'image_name':'light_pant', 'bbox': [], 'label': 'Pants', 'condifence': 0.9, 'color': 'silver', 'top_category': 'Bottom'
    }
    meta_data_list.append(elem1)
    meta_data_list.append(elem2)
    meta_data_list.append(elem3)
    meta_data_list.append(elem4)
    local_matrix, all_images = get_local_matrix(meta_data_list)
    print(local_matrix)
    print(all_images)