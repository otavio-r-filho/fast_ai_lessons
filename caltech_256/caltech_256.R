library(dplyr)
library(readr)
library(mlbench)
library(mxnet)

train.rec <- "../datasets/image_classification/caltech_256/processed_files/caltech256-train.rec"
train.lst <- "../datasets/image_classification/caltech_256/processed_files/caltech256-train.lst"
train.idx <- "../datasets/image_classification/caltech_256/processed_files/caltech256-train.idx"
val.rec <- "../datasets/image_classification/caltech_256/processed_files/caltech256-val.rec"
val.lst <- "../datasets/image_classification/caltech_256/processed_files/caltech256-val.lst"
val.idx <- "../datasets/image_classification/caltech_256/processed_files/caltech256-val.idx"

train_iter <- mx.io.ImageRecordIter(
  path.imglist = train.lst,
  path.imgrec = train.rec,
  path.imgidx = train.idx,
  data.shape = c(299,299,3),
  batch.size = 48,
  resize = 299
)

val_iter <- mx.io.ImageRecordIter(
  path.imglist = val.lst,
  path.imgrec = val.rec,
  path.imgidx = val.idx,
  data.shape = c(299,299,3),
  batch.size = 48,
  resize = 299
)

source("symbol_vgg.R")
nn_vgg <- get_symbol(num_classes = 256)

model <- mx.model.FeedForward.create(
  symbol = nn_vgg,
  X = train_iter,
  eval.data = val_iter,
  ctx = mx.cpu(),
  num.round = 50,
  initializer = mx.init.Xavier(),
  optimizer = "adam",
  learning.rate = 0.001,
  eval.metric = mx.metric.logloss,
  epoch.end.callback = mx.callback.save.checkpoint("symbol_vgg")
)
