# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 08:11:53 2019

@author: otavi
"""

from os import path, mkdir, system, remove
from glob import glob
import re
import numpy as np
import pandas as pd
from sys import platform

def get_species(x):
    return "cat" if str.isupper(x[0]) else "dog"

def remove_ext(x):
        tmp = re.sub("_([0-9]{1,})\.jpg", "", x)
        return tmp.replace("_", " ", 10)
    
def get_classid(img_file, class2id):
    return class2id[remove_ext(img_file)]

def assign_class_dict(img_list):    
    class_list = list(set(map(remove_ext, img_list)))
    class_list = sorted(class_list)
    
    class_ids = np.arange(1, len(class_list) + 1)
    
    class2id = dict(zip(class_list, class_ids))
    id2class = dict(zip(class_ids, class_list))
    
    class_id_list = [get_classid(img_file, class2id) for img_file in img_list]
    
    return class_id_list, class2id, id2class

def create_image_dataframe(img_path):
    img_list = list(map(path.basename, glob(''.join([img_path + "/*.jpg"]))))
    species_list = list(map(get_species, img_list))
    class_id_list, class2id, id2class = assign_class_dict(img_list)
    
    pet_df = pd.DataFrame({"class": class_id_list, "file": img_list, "species": species_list})
    pet_df.index += 1
    return pet_df, class2id, id2class

img_path = "../../datasets/image_classification/oxford-iiit-pet/images"
dataset_path = "../../datasets/image_classification/oxford-iiit-pet"
if platform == "win32":
    img_path = "../../../datasets/image_classification/oxford-iiit-pet/images"
    dataset_path = "../../../datasets/image_classification/oxford-iiit-pet"

    
pet_df, class2id, id2class = create_image_dataframe(img_path)

cmd_list = []
for k, v in id2class.items():
    class_dir = path.join(img_path, v)
    class_dir = ''.join(["\"", class_dir, "\""])
    cmd = ''.join(["mkdir ", class_dir, "\n"])
    if platform == "win32":
        cmd = cmd.replace("/", "\\", 50)
    cmd_list.append(cmd)
    class_fls = pet_df.loc[pet_df["class"] == k].loc[:, "file"].values
    for f in class_fls:
        cmd = ''.join(["mv ", path.join(img_path, f), " ", class_dir, "\n"])
        if platform == "win32":
            cmd = ''.join(["move ", path.join(img_path, f), " ", class_dir, "\n"])
            cmd = cmd.replace("/", "\\", 50)
        cmd_list.append(cmd)
        
if platform == "win32":
    f = open("commands.bat", "w")
    f.writelines(cmd_list)
    f.close()
    system("commands.bat")
else:
    f = open("commands.sh", "w")
    f.writelines(cmd_list)
    f.close()
    system("chmod 700 commands.sh")
    system("sh commands.sh")
