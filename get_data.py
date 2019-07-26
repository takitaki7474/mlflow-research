import pandas as pd
import cv2
import numpy as np
import os

def feature_table_sort(df,sort_column,ascending=True):
    df = df.sort_values(by=[sort_column], ascending=ascending)
    df = df.reset_index()

    return df

def over_search_img(df,img_num):
    images = list(df["img"][:img_num])

    return images

def under_search_img(df,img_num):
    images = list(df["img"][-img_num:])

    return images

def create_train_dataset(feature_table, TRAIN_FOLDER_NAME, image_set):
    train_dataset = []
    for image in image_set:
        label = feature_table[feature_table.img == image].label.values[0]
        filepath = os.path.join(TRAIN_FOLDER_NAME, str(label), image)
        train_dataset.append((filepath, np.int32(label)))

    return train_dataset

def dataset_conversion(dataset):
    img_data = []
    label_data = []

    for filepath, label in dataset:
        img = cv2.imread(filepath, 1)
        img = cv2.resize(img,(28,28))
        img_data.append(img)
        label_data.append(label)

    img_data = np.array(img_data).astype(np.float32)
    label_data = np.array(label_data).astype(np.int32)

    return img_data, label_data
