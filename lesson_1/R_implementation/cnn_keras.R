library(keras)
use_condaenv("dlpy36")

get_keras_model <- function(input_shape, num_classes) {
  inputs <- keras::layer_input(shape = input_shape)
  
  # Conv block 1
  conv_1_1 <- keras::layer_conv_2d(inputs, filters = 32, kernel_size = c(32,32), strides = c(1,1), activation = "relu", padding = "same", name = "conv_1_1")
  conv_1_2 <- keras::layer_conv_2d(conv_1_1, filters = 32, kernel_size = c(32,32), strides = c(1,1), activation = "relu", padding = "same", name = "conv_1_2")
  pool_1 <- keras::layer_max_pooling_2d(conv_1_2, pool_size = c(4,4), strides = c(2,2), padding = "valid", name = "pool_1")
  
  # Conv block 2
  #conv_2_1 <- keras::layer_conv_2d(pool_1, filters = 128, kernel_size = c(32,32), strides = c(2,2), activation = "relu", name = "conv_2_1")
  #conv_2_2 <- keras::layer_conv_2d(conv_2_1, filters = 128, kernel_size = c(32,32), strides = c(1,1), activation = "relu", name = "conv_2_2")
  #pool_2 <- keras::layer_max_pooling_2d(conv_2_2, pool_size = c(2,2), strides = c(2,2), name = "pool_2")
  
  # Conv block 3
  #conv_3_1 <- keras::layer_conv_2d(pool_2, filters = 256, kernel_size = c(16,16), strides = c(2,2), activation = "relu", name = "conv_3_1")
  #conv_3_2 <- keras::layer_conv_2d(conv_3_1, filters = 256, kernel_size = c(16,16), strides = c(1,1), activation = "relu", name = "conv_3_2")
  #pool_3 <- keras::layer_max_pooling_2d(conv_3_2, pool_size = c(2,2), strides = c(2,2), name = "pool_3")
  
  # Conv block 4
  #conv_4_1 <- keras::layer_conv_2d(pool_3, filters = 512, kernel_size = c(4,4), strides = c(2,2), activation = "relu", name = "conv_4_1")
  #conv_4_2 <- keras::layer_conv_2d(conv_4_1, filters = 512, kernel_size = c(4,4), strides = c(1,1), activation = "relu", name = "conv_4_2")
  
  #flat_pool <- keras::layer_global_average_pooling_2d(conv_4_2, name = "avg_global_pool")
  
  #fc_1 <- keras::layer_dense(flat_pool, units = 2048, activation = "relu", name = "fc_1")
  #do_1 <- keras::layer_dropout(fc_1, rate = 0.5, name = "do_1")
  
  #fc_2 <- keras::layer_dense(do_1, units = 128, activation = "relu", name = "fc_2")
  #do_2 <- keras::layer_dropout(fc_2, rate = 0.35, name = "do_2")
  
  #logits <- keras::layer_dense(do_2, units = num_classes, activation = None, name = "logits")
  
  #output <- keras::layer_activation(logits, activation = "softmax", name = "output")
  
  output <- pool_1
  
  model <- keras_model(inputs = inputs, outputs = output)
  
  return(model)
}

model <- get_keras_model(input_shape = c(256,256,3), num_classes = 37)
summary(model)
