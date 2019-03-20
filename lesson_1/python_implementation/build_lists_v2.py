# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 08:11:53 2019

@author: otavi
"""

from os import path
from glob import glob
import re

img_path = "../../datasets/image_classification/oxford-iiit-pet/images/*.jpg"

def get_species(x):
    return "cat" if str.isupper(x[0]) else "dog"

def get_class_list(img_list):
    def remove_ext(x):
        tmp = re.sub("_([0-9]{1,})\.jpg", "", x)
        tmp = tmp.replace("_", " ", 10)
        return tmp
    
    class_list = list(map(remove_ext, img_list))
    class_list = set(class_list)
    return class_list

img_list = list(map(path.basename, glob(img_path)))
img_list
species_list = list(map(get_species, img_list))
species_list

class_list = get_class_list(img_list)
class_list
