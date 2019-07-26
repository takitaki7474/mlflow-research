import pandas as pd

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
