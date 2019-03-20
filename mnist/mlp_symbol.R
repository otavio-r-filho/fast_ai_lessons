require(mlbench)
require(mxnet)

get_symbol <- function(num_classes) {
  mx.symbol.Variable(name = "data") %>%
    mx.symbol.FullyConnected(., num_hidden = 512, name = "fc_1") %>%
    mx.symbol.Activation(., act_type = "relu", name = "rel_fc_1") %>%
    mx.symbol.FullyConnected(., num_hidden = 256, name = "fc_2") %>%
    mx.symbol.Activation(., act_type = "relu", name = "rel_fc_2") %>%
    mx.symbol.FullyConnected(., num_hidden = 96, name = "fc_3") %>%
    mx.symbol.Activation(., act_type = "relu", name = "rel_fc_3") %>%
    mx.symbol.FullyConnected(., num_hidden = num_classes, name = "logits") %>%
    mx.symbol.SoftmaxOutput(., name = "output") -> softmax
  
  return(softmax)
}