library(keras)
use_condaenv("dlpy36")

get_keras_model <- function(input_shape, num_classes) {
    inputs <- keras::layer_input(shape = input_shape)

    conv_1_1 <- keras::layer_conv_2d(inputs, filters = 64, kernel_size = c(4,4), strides = c(1,1), activation = "relu", name = "conv_1_1")
    conv_1_2 <- keras::layer_conv_2d(conv_1_1, filters = 64, kernel_size = c(4,4), strides = c(1,1), activation = "relu", name = "conv_1_2")
    pool_1 <- keras::layer_max_pooling_2d(conv_1_2, pool_size = c(2,2), strides = c(2,2), name = "pool_1")
    
    conv_2_1 <- keras::layer_conv_2d(pool_1, filters = 128, kernel_size = c(3,3), strides = c(2,2), activation = "relu", name = "conv_2_1")
    conv_2_2 <- keras::layer_conv_2d(conv_2_1, filters = 128, kernel_size = c(3,3), strides = c(1,1), activation = "relu", name = "conv_2_2")
    pool_2 <- keras::layer_max_pooling_2d(conv_2_2, pool_size = c(2,2), strides = c(2,2), name = "pool_2")
    
    conv_3_1 <- keras::layer_conv_2d(pool_2, filters = 512, kernel_size = c(3,3), strides = c(2,2), activation = "relu", name = "conv_3_1")
    conv_3_2 <- keras::layer_conv_2d(conv_3_1, filters = 512, kernel_size = c(3,3), strides = c(1,1), activation = "relu", name = "conv_3_2")
    pool_3 <- keras::layer_max_pooling_2d(conv_3_2, pool_size = c(2,2), strides = c(2,2), name = "pool_3")
    
    flatten <- keras::layer_flatten(pool_3, name = "flatten")
    
    fc_1 <- keras::layer_dense(flatten, units = 1024, activation = "relu", name = "fc_1")
    fc_2 <- keras::layer_dense(fc_1, units = 128, activation = "relu", name = "fc_2")
    
    logits <- keras::layer_dense(fc_2, units = num_classes, name = "logits")

    output <- keras::layer_activation(logits, activation = "softmax")
    
    model <- keras_model(inputs = inputs, outputs = output)
  
    return(model)
}

model <- get_keras_model(input_shape = c(256,256,3), num_classes = 37)
summary(model)