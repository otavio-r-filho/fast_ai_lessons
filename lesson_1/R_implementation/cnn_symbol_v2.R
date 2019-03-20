require(mxnet)
require(magrittr)
require(dplyr)
if(!require(mlbench)) {
  install.packages("mlbench")
}

get_symbol <- function(num_classes) {
  # Input
  mx.symbol.Variable(name = "data") %>%
    # Conv block 1
    mx.symbol.Convolution(., kernel = c(4,4), stride = c(1,1), num_filter = 64, name = "conv_1_1") %>%
    mx.symbol.Activation(., act_type = "relu", name = "act_relu_1_1") %>%
    mx.symbol.Convolution(., kernel = c(4,4), stride = c(1,1), num_filter = 64, name = "conv_1_2") %>%
    mx.symbol.Activation(., act_type = "relu", name = "act_relu_1_2") %>%
    mx.symbol.Pooling(., kernel = c(2,2), stride = c(2,2), pool_type = "max", name = "pool_1") %>%
    # Conv block 2
    mx.symbol.Convolution(., kernel = c(3,3), stride = c(2,2), num_filter = 128, name = "conv_1_1") %>%
    mx.symbol.Activation(., act_type = "relu", name = "act_relu_2_1") %>%
    mx.symbol.Convolution(., kernel = c(3,3), stride = c(1,1), num_filter = 128, name = "conv_2_2") %>%
    mx.symbol.Activation(., act_type = "relu", name = "act_relu_2_2") %>%
    mx.symbol.Pooling(., kernel = c(2,2), stride = c(2,2), pool_type = "max", name = "pool_2") %>%
    # Conv block 3
    mx.symbol.Convolution(., kernel = c(3,3), stride = c(2,2), num_filter = 512, name = "conv_3_1") %>%
    mx.symbol.Activation(., act_type = "relu", name = "act_relu_3_1") %>%
    mx.symbol.Convolution(., kernel = c(3,3), stride = c(1,1), num_filter = 512, name = "conv_3_2") %>%
    mx.symbol.Activation(., act_type = "relu", name = "act_relu_3_2") %>%
    mx.symbol.Pooling(., kernel = c(2,2), stride = c(2,2), pool_type = "max", name = "pool_3") %>%
    # Flatten
    mx.symbol.Flatten(., name = "flatten") %>%
    # Fully connected 1
    mx.symbol.FullyConnected(., num_hidden = 1024, name = "fc_1") %>%
    mx.symbol.Activation(., act_type = "relu", name = "act_fc_1") %>%
    # Fully connected 2
    mx.symbol.FullyConnected(., num_hidden = 128, name = "fc_2") %>%
    mx.symbol.Activation(., act_type = "relu", name = "act_fc_2") %>%
    # Output block
    mx.symbol.FullyConnected(., num_hidden = num_classes, name = "logits") %>%
    mx.symbol.SoftmaxOutput(., name = "output") -> output
  return(output)
}