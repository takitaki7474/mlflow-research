import os
import numpy as np
import cv2
import pandas as pd
import torch
import torch.utils.data
import processing_table
import get_data
import mlflow_run

TRAIN_FOLDER_NAME = "./train_img"
IMG_NUM_BY_CLASS = 300
EPOCH = 50


if __name__=="__main__":

    feature_table_path = "./feature/feature_table.pkl"
    feature_table = pd.read_pickle(feature_table_path)

    ft = processing_table.ProcessFeatureTable(feature_table)

    devide_df = ft.table_division()

    image_set = []
    for df in devide_df:
        image_set += get_data.over_search_img(df, IMG_NUM_BY_CLASS)

    train_dataset = get_data.create_train_dataset(feature_table, TRAIN_FOLDER_NAME, image_set)
    img_data, label_data = get_data.dataset_conversion(train_dataset)

    train_data = []
    for x_train, y_train in zip(img_data, label_data):
        x_train = x_train.reshape(3,28,28)
        train_data.append((x_train, y_train))

    dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

    mlflow_run.train(EPOCH, dataloader)
