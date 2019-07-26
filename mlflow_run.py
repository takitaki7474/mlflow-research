import os
import numpy as np
import cv2
import pandas as pd
import torch
import torch.utils.data
import processing_table
import get_data_from_table

TRAIN_FOLDER_NAME = "./train_img"
IMG_NUM_BY_CLASS = 300

if __name__=="__main__":

    feature_table_path = "./feature/feature_table.pkl"
    feature_table = pd.read_pickle(feature_table_path)

    ft = processing_table.ProcessFeatureTable(feature_table)

    devide_df = ft.table_division()

    image_set = []
    for df in devide_df:
        image_set += over_search_img(df, img_num_by_class)
