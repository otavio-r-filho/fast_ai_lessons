import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lst_path = "../../datasets/image_classification/oxford-iiit-pet/annotations/list.txt"

img_list = pd.read_csv(lst_path, sep = ' ', comment = "#", header = None,
                       names = ["file", "class id", "species", "breed id"])
img_list = img_list[["class id", "file", "species"]]

img_list.loc[:, "species"].replace(1, "cat", inplace = True)
img_list.loc[:, "species"].replace(2, "dog", inplace = True)

class_dict = img_list

img_list.loc[:, "file"] = img_list.loc[:, "file"].apply(lambda x: x + ".jpg")
img_list.loc[:,["class id", "file"]].to_csv("img.list", sep = '\t', header = False)

img_list.head()

class_dict.loc[:, "file"].replace("_([0-9]{1,}\.jpg)", "", regex = True, inplace = True)
class_dict.loc[:, "file"].replace("_", " ", regex = True, inplace = True)
class_dict.drop_duplicates(keep = "first", inplace = True)
class_dict.rename(columns  = {"class id": "class", "file": "breed"}, inplace = True)
class_dict.set_index(keys = "class", inplace = True)
class_dict.head()
class_dict = class_dict.to_dict("index")
