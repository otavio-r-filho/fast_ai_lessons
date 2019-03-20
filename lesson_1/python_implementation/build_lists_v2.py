# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 08:11:53 2019

@author: otavi
"""

from os import path
from glob import glob
import re
import numpy as np
import pandas as pd

img_path = "../../datasets/image_classification/oxford-iiit-pet/images/*.jpg"

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

img_list = list(map(path.basename, glob(img_path)))
species_list = list(map(get_species, img_list))
class_id_list, class2id, id2class = assign_class_dict(img_list)

pet_df = pd.DataFrame({"class": class_id_list, "file": img_list, "species": species_list})
pet_df.head(10)
