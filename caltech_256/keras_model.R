library(dplyr)
library(tidyr)
library(stringr)
library(keras)
use_condaenv("dlpy36-mxnet")

build_conv_block <- function(block_input, block_layer, kernel_size, num_filters, activation = "relu") {
  conv_1_name <- str_c("conv", block_layer, "1", sep = "_")
  conv_2_name <- str_c("conv", block_layer, "2", sep = "_")
  pool_name <- str_c("pool", block_layer, sep = "_")
  
  conv_1 <- keras::layer_conv_2d(block_input, filters = num_filters, kernel_size = kernel_size, activation = activation, strides = c(1,1), name = conv_1_name)
  conv_2 <- keras::layer_conv_2d(conv_1, filter = num_filters, kernel_size = kernel_size, activation = activation, strides = c(1,1), name = conv_2_name)
  pool <- keras::layer_max_pooling_2d(conv_2, pool_size = c(2,2), strides = c(2,2), name = pool_name)
}

get_model <- function(num_classes, input_shape) {
  inputs <- keras::layer_input(shape = input_shape, name = "inputs")
  
  conv_block_1 <- build_conv_block(inputs, block_layer = 1, kernel_size = c(4,4), num_filters = 32, activation = "relu")
  conv_block_2 <- build_conv_block(conv_block_1, block_layer = 2, kernel_size = c(4,4), num_filters = 64, activation = "relu")
  conv_block_3 <- build_conv_block(conv_block_2, block_layer = 3, kernel_size = c(3,3), num_filters = 256, activation = "relu")
  #conv_block_4 <- build_conv_block(conv_block_3, block_layer = 4, kernel_size = c(3,3), num_filters = 512, activation = "relu")
  
  flattened <- keras::layer_flatten(conv_block_3, name = "flatten")
  
  dense_1 <- keras::layer_dense(flattened, units = 2048, activation = "relu", name = "fc_1")
  dense_2 <- keras::layer_dense(dense_1, units = 512, activation = "relu", name = "fc_2")
  
  outputs <- keras::layer_dense(dense_2, units = num_classes, activation = "softmax", name = "outputs")
  
  model <- keras_model(inputs = inputs, outputs = outputs)
  
  summary(model)
  return(model)
}

model <- get_model(num_classes = 257, input_shape = c(299,299,3))
batch_size = 24

train_datagen <- keras::image_data_generator(rescale = 1./255, validation_split = 0.3)
dataset_dir <- "../datasets/image_classification/caltech_256/256_ObjectCategories"
train_generator <- keras::flow_images_from_directory(directory = dataset_dir,
                                                     generator = train_datagen,
                                                     target_size = c(299,299),
                                                     class_mode = "categorical",
                                                     batch_size = batch_size)
model %>% compile(optimizer = "adam",
                  loss = 'categorical_crossentropy',
                  metrics = c('accuracy'))

steps_epoch = round((15420 + 15187) / batch_size, digits = 0)
model %>% fit_generator(.,
                        generator = train_generator,
                        epochs = 120,
                        steps_per_epoch = steps_epoch,
                        verbose = 1)
