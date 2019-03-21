from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D

def build_conv_block(block_input, block_layer, kernel_size, num_filters, activation = "relu"):
    conv_1_name = '_'.join(["conv", block_layer, "1"])
    conv_2_name = '_'.join(["conv", block_layer, "2"])
    pool_name = '_'.join(["pool", block_layer])

    conv_1 = Conv2D(filters = num_filters, kernel_size = kernel_size, activation = activation, strides = (1,1), name = conv_1_name) (block_input)
    conv_2 = Conv2D(filters = num_filters, kernel_size = kernel_size, activation = activation, strides = (2,2), name = conv_2_name) (conv_1)
    pool = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = pool_name)(conv_2)

    return pool

def get_model(num_classes, input_shape):
  inputs = Input(shape = input_shape, name = "inputs")
  
  conv_block_1 = build_conv_block(inputs, block_layer = "1", kernel_size = (4,4), num_filters = 32, activation = "relu")
  conv_block_2 = build_conv_block(conv_block_1, block_layer = "2", kernel_size = (4,4), num_filters = 64, activation = "relu")
  conv_block_3 = build_conv_block(conv_block_2, block_layer = "3", kernel_size = (3,3), num_filters = 256, activation = "relu")
#   conv_block_4 = build_conv_block(conv_block_3, block_layer = "4", kernel_size = (3,3), num_filters = 512, activation = "relu")
  
  flattened = Flatten(name = "flatten")(conv_block_3)
  
  dense_1 = Dense(units = 2048, activation = "relu", name = "fc_1")(flattened)
  dense_2 = Dense(units = 512, activation = "relu", name = "fc_2")(dense_1)
  
  outputs = Dense(units = num_classes, activation = "softmax", name = "outputs")(dense_2)
  
  model = Model(inputs = inputs, outputs = outputs)
  
  print(model.summary())
  return(model)

if __name__ == "__main__":
    model = get_model(256, (299,299,3))