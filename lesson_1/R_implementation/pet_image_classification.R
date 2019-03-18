library(mxnet)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(readr)

# Loaging labels
labels <- read_table2("../../datasets/image_classification/oxford-iiit-pet/annotations/list.txt", 
                      col_names = FALSE, comment = "#")
labels <- labels[,1:3]
names(labels) <- c("image_file", "class_id", "species")

# Preparing the labels dictionary
labels_dict <- labels %>%
               mutate(image_file = str_replace(image_file, "_([0-9]{1,})", "")) %>% 
               mutate(image_file = str_replace_all(image_file, "_", " ")) %>%
               mutate(species = if_else(species == 1, "cat", "dog")) %>%
               mutate(breed_id = NULL) %>%
               unique(.)
names(labels_dict) <- c("breed", "class", "species")
#View(labels_dict)

labels <- labels %>%
          mutate(index = row.names(labels)) %>%
          mutate(image_file = str_c(image_file, ".jpg", sep = ""))
                    
labels <- labels[c("index", "class_id", "image_file")]
#View(labels)

source("aux_functions.R")
input_list <- split_dataset(labels, 0.7, 0.15)
write_delim(input_list$train, "train_list", delim = " \t ", col_names = F)
write_delim(input_list$val, "val_list", delim = " \t ", col_names = F)
write_delim(input_list$test, "test_list", delim = " \t ", col_names = F)

View(input_list$train)

source("cnn_symbol.R")
symbol <- get_symbol(nrow(labels_dict))

train_bin <- "../../datasets/image_classification/oxford-iiit-pet/processed_files/train.bin"
val_bin <- "../../datasets/image_classification/oxford-iiit-pet/processed_files/val.bin"
test_bin <- "../../datasets/image_classification/oxford-iiit-pet/processed_files/val.bin"

im2rec(image_lst = "train_list",
       root = "../../datasets/image_classification/oxford-iiit-pet/images/",
       output_rec = train_bin)

im2rec(image_lst = "val_list",
       root = "../../datasets/image_classification/oxford-iiit-pet/images/",
       output_rec = val_bin)

im2rec(image_lst = "test_list",
       root = "../../datasets/image_classification/oxford-iiit-pet/images/",
       output_rec = test_bin)

train <- mx.io.ImageRecordIter(path.imgrec = train_bin,
                               batch.size = 32,
                               data.shape = c(256, 256, 3))

val <- mx.io.ImageRecordIter(path.imgrec = val_bin,
                             batch.size = 32,
                             data.shape = c(256, 256, 3))

model <- mx.model.FeedForward.create(
  X = train,
  eval.data = val,
  ctx = mx.gpu(),
  symbol = symbol,
  eval.metric = mx.metric.top_k_accuracy,
  num.round = 50,
  batch.end.callback = mx.callback.log.train.metric(10),
  epoch.end.callback = mx.callback.save.checkpoint("dg_classifier", period = 10),
  verbose = T,
  initializer = mx.init.Xavier()
)
