#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:22:47 2019

@author: otavio
"""

from glob import glob
from os import path, mkdir, system, remove
import pandas as pd
import re
from random import shuffle
from sys import platform

def os_cmd(cmd):
    '''
    Function to return basic commands based on the os
    '''
    commands = {
            "mkdir": {"linux": "mkdir", "win32": "mkdir"},
            "mv": {"linux": "mv", "win32": "move"},
            "rm -rvf": {}
    }

def remove_ext(x):
    '''
    Function to remove the extension of the files
    '''
    return re.sub("_([0-9]{1,})\.jpg", "", x)

def split_images(class_name, dataset_path, class_img_list, val_split, test_split=0.0):
    '''
    1 - Create the paths for the class (i.e. train/val/test)
    2 - Add the creation commands of the paths to the command list
    3 - Set the ranges of the datasets inside the list
    4 - 
    '''
    make_test_set = test_split > 0.0
    class_paths = [
            path.join(dataset_path, "train", class_name),
            path.join(dataset_path, "val", class_name)
    ]
    
    if make_test_set: class_paths.appent(path.join(dataset_path, "test", class_name))
    
    cmd_list = (" ".join(["mkdir", cl_path]) for cl_path in class_paths)
    
    class_ranges = (
            
    )


dataset_path = "../../datasets/image_classification/oxford-iiit-pet/images"
if platform == 'win32': dataset_path = "..\..\..\datasets\image_classification\oxford-iiit-pet\images"

train_dataset_path = path.join(dataset_path, "train")
val_dataset_path = path.join(dataset_path, "val")
test_dataset_path = path.join(dataset_path, "test")

img_list = glob(''.join([dataset_path, "/*.jpg"]))

val_split = 0.15
test_split = 0.15

class_name = set(map(remove_ext, img_list))
class_name = list(map(path.basename, class_name))

img_df = pd.DataFrame({"image": img_list})

# Adding initial command list to create the subdirs
cmd_list = [
        " ".join(["mkdir", train_dataset_path, "\n"]),
        " ".join(["mkdir", val_dataset_path, "\n"]),
        " ".join(["mkdir", test_dataset_path, "\n"])
        ]

for prefix in class_name:
    # Loading images according to prefix
    class_img_list = img_df.loc["image"].str.contains(prefix).values
