#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:59:09 2019

@author: otavio
"""

from glob import glob
from os import path, mkdir, system, remove
import pandas as pd
import re
from random import shuffle

def remove_ext(x):
    return re.sub("_([0-9]{1,})\.jpg", "", x)
        

dataset_path = "../../datasets/image_classification/oxford-iiit-pet/images"
train_dataset_path = path.join(dataset_path, "train")
val_dataset_path = path.join(dataset_path, "val")
test_dataset_path = path.join(dataset_path, "test")

img_list = glob(''.join([dataset_path, "/*.jpg"]))

val_split = 0.15
test_split = 0.15

img_prefix = set(map(remove_ext, img_list))
img_prefix = list(map(path.basename, img_prefix))

img_df = pd.DataFrame({"image": img_list})

cmd_list = [
        " ".join(["mkdir", train_dataset_path, "\n"]),
        " ".join(["mkdir", val_dataset_path, "\n"]),
        " ".join(["mkdir", test_dataset_path, "\n"])
        ]

for prefix in img_prefix:
    # Loading images according to prefix
    class_img_list = img_df.loc["image"].str.contains(prefix).values
    class_list_len = len(class_img_list)
    
    # Image quantities
    train_img_qtd = round(class_list_len * (1-val_split-test_split), 0)
    val_img_qtd = round(class_list_len * val_split, 0)
    test_img_qtd = class_list_len - train_img_qtd - val_img_qtd
    
    # Adding dir creation to the commando list
    # Train class dir
    prefix_path = path.join(train_dataset_path, prefix)
    cmd = " ".join(["mkdir", prefix_path, "\n"])
    cmd_list.append(cmd)
    # Val class dir
    prefix_path = path.join(train_dataset_path, prefix)
    cmd = " ".join(["mkdir", prefix_path, "\n"])
    cmd_list.append(cmd)
    # Test class dir
    prefix_path = path.join(train_dataset_path, prefix)
    cmd = " ".join(["mkdir", prefix_path, "\n"])
    cmd_list.append(cmd)
    
    
prefix = img_prefix[10]
class_img_list = img_df.loc[img_df["image"].str.contains(prefix)].values
train_img_qtd = round(len(class_img_list) * (1-val_split-test_split), 0)
val_img_qtd = round(len(class_img_list) * val_split, 0)
test_img_qtd = len(class_img_list) - train_img_qtd - val_img_qtd

cmd = " ".join(["mkdir", ])
cmd_list.append()

cmd_file = "group_files.sh"
f = open(cmd_file, "w")
f.writelines(cmd_list)
f.close()

system(" ".join(["sh", cmd_file]))
