#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:22:47 2019

@author: otavio
"""

from glob import glob
from os import path, system, remove, walk, scandir
import pandas as pd
import re
from time import time
import numpy as np
from numpy.random import shuffle, seed, randint, rand
from sys import platform
from sklearn.preprocessing import OneHotEncoder
import subprocess
from PIL import Image

def os_cmd(cmd):
    '''
    Function to return basic commands based on the os
    '''
    commands = {
            "mkdir": {"linux": "mkdir", "win32": "mkdir"},
            "mv": {"linux": "mv", "win32": "move"},
            "rm -rvf": {}
    }
    return commands[cmd][platform]    

def get_hex_key(key_len = 10):
    '''
    Function to generate random hexadecimal keys
    
    '''
    seed(int(round(np.sqrt(time()) * rand() * 10, 0)))
    hex_key = list(map(hex, randint(0,16,key_len)))
    hex_key = list(map(re.sub, key_len*["0x"], key_len*[""], hex_key))
    
    return ''.join(hex_key)

def remove_ext(x):
    '''
    Function to remove the extension of the files
    '''
    return re.sub("_([0-9]{1,})\.jpg", "", x)

def split_class(class_name, dataset_path, class_img_list, val_split, test_split):
    '''
    1 - Create the paths for the class (i.e. train/val/test)
    2 - Add the creation commands of the paths to the command list
    3 - Set the ranges of the datasets inside the list
    4 - 
    '''
    make_test_set = test_split > 0.0
    
    train_img_qtd = int(round(len(class_img_list) * (1-val_split-test_split), 0))
    val_img_qtd = int(round(len(class_img_list) * val_split, 0))
    
    class_paths = [
            path.join(dataset_path, "train", class_name),
            path.join(dataset_path, "val", class_name)
    ]
    
    class_ranges = [
            (0, train_img_qtd),
            (train_img_qtd, train_img_qtd+val_img_qtd)
    ]
    
    if make_test_set:
        class_paths.append(path.join(dataset_path, "test", class_name))
        class_ranges.append((train_img_qtd+val_img_qtd, len(class_img_list)))
    
    cmd_list = [" ".join([os_cmd("mkdir"), cl_path, "\n"]) for cl_path in class_paths]
    
    for p, r, in zip(class_paths, class_ranges):
        cmd_list += [" ".join([os_cmd("mv"), img_p, p, "\n"]) for img_p in class_img_list[r[0]:r[1]] ]
    
    return cmd_list

def split_dataset(dataset_path, val_split, test_split = 0.0):
    '''
    '''
    train_dataset_path = path.join(dataset_path, "train")
    val_dataset_path = path.join(dataset_path, "val")
    test_dataset_path = path.join(dataset_path, "test")
    
    img_list = glob(''.join([dataset_path, "/*.jpg"]))
    
    class_names = set(map(remove_ext, img_list))
    class_names = list(map(path.basename, class_names))
    
    img_df = pd.DataFrame({"image": img_list})
    
    # Adding initial command list to create the subdirs
    cmd_list = [
            " ".join(["mkdir", train_dataset_path, "\n"]),
            " ".join(["mkdir", val_dataset_path, "\n"]),
            " ".join(["mkdir", test_dataset_path, "\n"])
            ]
    
    for class_name in class_names:
        # Loading images according to prefix
        class_img_list = img_df.loc[img_df["image"].str.contains(class_name)].values.squeeze()
        shuffle(class_img_list)
        cmd_list += split_class(class_name, dataset_path, class_img_list, val_split, test_split)
    
    fname = ''.join([get_hex_key(21), ".sh"])
    if(platform == "win32"): fname = ''.join([get_hex_key(21), ".bat"])
    
    fhandle = open(fname, "w")
    fhandle.writelines(cmd_list)
    fhandle.close()
    
    if platform == "win32":
        process_out = subprocess.run([fname], stderr = subprocess.STDOUT).returncode
        process_failed = process_out.returncode == 0        
    else:
        process_out1 = subprocess.run(["chmod", "700", fname], stderr = subprocess.STDOUT)
        process_out2 = subprocess.run(["sh", fname], stderr = subprocess.STDOUT)
        process_failed = (process_out1 == 0) and (process_out2 == 0)
        
    
    if process_failed:
        print("Something went wrong!")
        print(process_out.stderr)
    else:
        print("Success!")
    
    remove(fname)   

def load_dataset(dataset_path, shuffle_instances = True):
    '''
    Function to load 
    '''
    class_names = []
    class_files = []
    
    for root, dirs, files in walk(test_dataset_path):
        for f in files:
            class_names.append(path.basename(root))
            class_files.append(f)
            
    class_names = np.array(class_names)
    class_files = np.array(class_files)
    onehot_labels = OneHotEncoder(sparse = False).fit_transform(class_names.reshape((-1,1)))
    
    if shuffle_instances:
        shuffled_idx = np.arange(class_names.shape[0])
        np.random.shuffle(shuffled_idx)
        
        class_names = class_names[shuffled_idx]
        class_files = class_files[shuffled_idx]
        onehot_labels = onehot_labels[shuffled_idx]
        
    return class_names, class_files, onehot_labels

dataset_path = "../../datasets/image_classification/oxford-iiit-pet/images"
if platform == 'win32':
    dataset_path = "..\..\..\datasets\image_classification\oxford-iiit-pet\images"
train_dataset_path = path.join(dataset_path, "train")
val_dataset_path = path.join(dataset_path, "val")
test_dataset_path = path.join(dataset_path, "test")

val_split = 0.15
test_split = 0.15
split_dataset(dataset_path, val_split, test_split)
load_dataset(path.join(dataset_path, "test"))


