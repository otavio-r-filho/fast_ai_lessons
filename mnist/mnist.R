require(dplyr)
require(readr)

train_dataset_file <- "../datasets/image_classification/kaggle_mnist/train.csv"
test_dataset_file <- "../datasets/image_classification/kaggle_mnist/test.csv"

# Loading the dataset
train <- read_csv(train_dataset_file)
train <- data.matrix(train)

# Setting the split ratio
val_split <- round(0.3 * nrow(train), digits = 0)

# Splitting the dataset
val_x <- train[1:val_split,-1] / 255
val_y <- train[1:val_split,1]
train_x <- train[-(1:val_split),-1] / 255
train_y <- train[-(1:val_split),1]

# Loading the symbol graph
source("mlp_symbol.R")
nn_model <- get_symbol(10)

# Visualizing the graph
graph.viz(nn_model)

model <- mx.model.FeedForward.create(
  symbol = nn_model,
  X = train_x,
  y = train_y,
  ctx = mx.gpu(),
  eval.data = list(data = val_x, label = val_y),
  learning.rate = 0.01,
  eval.metric = mx.metric.top_k_accuracy,
  array.layout = "rowmajor",
  array.batch.size = 96,
  num.round = 50,
  epoch.end.callback = mx.callback.log.train.metric(50),
  batch.end.callback = mx.callback.log.speedometer(96)
)
