#!/usr/bin/env python


import argparse
import pandas as pd
import os
import json
import numpy as np
import pickle
from tqdm import tqdm


def create_img2idx(train_json_path, val_json_path, out_json_path):
    with open(train_json_path, 'rb') as f:
            data = json.load(f)
    train = pd.DataFrame(data)
    train_en = train[train['q_lang']=="en"]
    with open(val_json_path, 'rb') as f:
            data = json.load(f)
    val =  pd.DataFrame(data)
    val_en = val[val['q_lang']=="en"]
    img2idx = {}
    df = train_en.append(val_en)
    df_imgs = df['image_name'].unique().tolist()

    for i, row in tqdm(df.iterrows()):
        image_name = row['image_name']
        img_id = df_imgs.index(image_name)  # starts from 0
        if image_name not in img2idx:
            img2idx[image_name] = img_id
        else:
            assert img2idx[image_name] == img_id

    with open(out_json_path, 'w') as f:
        json.dump(img2idx, f)


if __name__ == "__main__":

    train_path = r'your_path.json'
    val_path = r'your_path.json'
    out_path = r'your_path.json'
    create_img2idx(train_path, val_path, out_path)
