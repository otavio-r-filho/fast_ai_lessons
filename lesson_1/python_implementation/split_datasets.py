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
from sys import platform

def os_cmd(cmd):
    commands = {
            "mv": {"linux": "mv", "win32": "move"},
            "cp": {"linux": "cp", "win32": "copy"},
            "mkdir": {"linux": "mkdir", "win32": "mkdir"}
    }
    
    return commands[cmd][platform]

def remove_ext(x):
    '''
    Function to remove the extension of the image files,
    this function also removes the number of examples.
    [class]_[example num].jpg
    '''
    return re.sub("_([0-9]{1,})\.jpg", "", x)

def add_prefix_path_cmd(class_name, dataset_path, make_test_path = True):
    '''
    Function to create the class directory in the trianing, validation and test directories
    '''
    class_paths = []
    path_type = ["train", "val"]
    if make_test_path: path_type.append("test")
    
    for pt in path_type:
        pt = path.join(dataset_path, pt)
        prefix_path = path.join(dataset_path, prefix)
        cmd = " ".join([os_cmd("mkdir"), prefix_path])
        cmd_list.append(cmd)
        
    return cmd_list

def add_split_images(cmd_list, split_range, set_type, dataset_path):
    '''
    This function takes 
    '''
    set_path = path.join(data)
    

def split_images(img_list, prefix, dataset_path, val_split, test_split = 0.0):
    '''
    Function to split the images of a class into training, validation
    and testing
    '''
    make_test_list = test_split > 0.0

    # Image quantities and ranges
    train_img_qtd = round(class_list_len * (1-val_split-test_split), 0)
    train_range = (train_img_qtd, train_img_qtd+val_img_qtd)

    val_img_qtd = round(class_list_len * val_split, 0)
    val_range = (train_img_qtd, train_img_qtd+val_img_qtd)
    
    # Defining the ranges of the lists by type of set
    if make_test_list:
        test_img_qtd = class_list_len - train_img_qtd - val_img_qtd
        path_type["test"] = (train_img_qtd+val_img_qtd, train_img_qtd+val_img_qtd+test_img_qtd)
    
    for pt in path_type:
        

dataset_path = "../../datasets/image_classification/oxford-iiit-pet/images"
train_dataset_path = path.join(dataset_path, "train")
val_dataset_path = path.join(dataset_path, "val")
test_dataset_path = path.join(dataset_path, "test")

img_list = glob(''.join([dataset_path, "/*.jpg"]))

val_split = 0.15
test_split = 0.15
make_test_path = test_split > 0.0

class_names = set(map(remove_ext, img_list))
class_names = list(map(path.basename, class_names))

img_df = pd.DataFrame({"image": img_list})

cmd_list = [
        " ".join([os_cmd("mkdir"), train_dataset_path, "\n"]),
        " ".join([os_cmd("mkdir"), val_dataset_path, "\n"]),
        " ".join([os_cmd("mkdir"), test_dataset_path, "\n"])
        ]

for prefix in img_prefix:
    # Loading images according to prefix
    class_img_list = img_df.loc["image"].str.contains(prefix).values
    class_list_len = len(class_img_list)
    
    # Adding directory creation for the training, validation and testing datasets
    cmd_list = add_prefix_path_cmd(prefix, dataset_path, cmd_list, make_test_path)
    
    
    
    
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
