require(dplyr)
require(tidyr)

split_dataset <- function(ds_list, train_split, val_split) {
  fcol = ncol(ds_list)
  
  # Add sum to 1 assertion
  train_list <- sample_frac(ds_list, train_split)
  rem_list <- ds_list %>% filter(! .[[fcol]] %in% train_list[[fcol]])
  
  val_list <- sample_n(rem_list, round(val_split * nrow(ds_list), digit = 0))
  rem_list <- rem_list %>% filter(! .[[fcol]] %in% val_list[[fcol]])
  
  if(train_split + val_split < 1.0){
    ret = list("train" = train_list, "val" = val_list, "test" = rem_list)
  } else {
    ret = list("train" = train_list, "val" = val_list)
  }
}

get_iterators <- function(input_list, input_shape) {
  train_it = mx.io.
}
