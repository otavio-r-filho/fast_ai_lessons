# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 08:11:53 2019

@author: otavi
"""

from os import path
from glob import glob

img_path = "../../datasets/image_classification/oxford-iiit-pet/images/*.jpg"

def get_species(x):
    return "cat" if str.isupper(x[0]) else "dog"

img_list = list(map(path.basename, glob(img_path)))
img_list
species_list = list(map(get_species, img_list))
species_list
