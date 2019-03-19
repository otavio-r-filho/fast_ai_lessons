require(mxnet)
if(!require(mlbench)){
  install.packages('mlbench')
}

get_symbol <- function(num_classes) {
  # Input
  input_data = mx.symbol.Variable(name = "data")
  
  # Convultional layer 1 | Input size [ 256, 256, 3]
  pad_size <- 32 / 2
  conv_1_1 = mx.symbol.Convolution(data = input_data, kernel = c(32,32), stride = c(1,1), num_filter = 32, pad = c(pad_size, pad_size), name = "conv_1_1")
  conv_1_1 = mx.symbol.Activation(data = conv_1_1, act_type = "relu", name = "act_relu_1_1")
  pad_size = pad_size - 1
  conv_1_2 = mx.symbol.Convolution(data = conv_1_1, kernel = c(32,32), stride = c(1,1), num_filter = 30, pad = c(pad_size, pad_size), name = "conv_1_2")
  conv_1_2 = mx.symbol.Activation(data = conv_1_2, act_type = "relu", name = "act_relu_1_2")
  pool_1 = mx.symbol.Pooling(data = conv_1_2, kernel = c(2,2), stride = c(2,2), pool_type = "max", name = "pool_1")
  
  # Convultional layer 2 | Input shape [128, 128, 32]
  pad_size = 16 / 2
  conv_2_1 = mx.symbol.Convolution(data = pool_1, kernel = c(16, 16), stride = c(1,1), num_filter = 128, pad = c(pad_size, pad_size), name = "conv_1_1")
  conv_2_1 = mx.symbol.Activation(data = conv_2_1, act_type = "relu", name = "act_relu_2_1")
  pad_size = pad_size - 1
  conv_2_2 = mx.symbol.Convolution(data = conv_2_1, kernel = c(16,16), stride = c(1,1), num_filter = 125, pad = c(pad_size, pad_size), name = "conv_2_2")
  conv_2_2 = mx.symbol.Activation(data = conv_2_2, act_type = "relu", name = "act_relu_2_2")
  pool_2 = mx.symbol.Pooling(data = conv_2_2, kernel = c(2,2), stride = c(2,2), pool_type = "max", name = "pool_2")
  
  # Convultional layer 3 | Input shape [64, 64, 128]
  pad_size = 4 / 2
  conv_3_1 = mx.symbol.Convolution(data = pool_2, kernel = c(4,4), stride = c(1,1), num_filter = 256, pad = c(pad_size, pad_size), name = "conv_3_1")
  conv_3_1 = mx.symbol.Activation(data = conv_3_1, act_type = "relu", name = "act_relu_3_1")
  pad_size = pad_size - 1
  conv_3_2 = mx.symbol.Convolution(data = conv_3_1, kernel = c(4,4), stride = c(1,1), num_filter = 250, pad = c(pad_size, pad_size), name = "conv_3_2")
  conv_3_2 = mx.symbol.Activation(data = conv_3_2, act_type = "relu", name = "act_relu_3_2")
  
  flat_pool = mx.symbol.Pooling(data = conv_3_2, pool_type = "avg", global_pool = T, name = "avg_global_pool")
  
  fc_1 = mx.symbol.FullyConnected(data = flat_pool, num_hidden = 512, name = "fc_1")
  fc_1 = mx.symbol.Activation(data = fc_1, act_type = "relu", name = "fc_1_relu")
  dout_1 = mx.symbol.Dropout(data = fc_1, p = 0.5, name = "dout_1")
  
  fc_2 = mx.symbol.FullyConnected(data = dout_1, num_hidden = 64, name = "fc_2")
  fc_2 = mx.symbol.Activation(data = fc_2, act_type = "relu", name = "fc_2_relu")
  dout_2 = mx.symbol.Dropout(data = fc_2, p = 0.15, name = "dout_2")
  
  logits = mx.symbol.FullyConnected(data = dout_2, num_hidden = num_classes, name = "logits")
  
  softmax = mx.symbol.SoftmaxOutput(data = logits, name = "softmax")
  
  return(softmax)
}
