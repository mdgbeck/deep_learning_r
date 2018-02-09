library(keras)
library(tidyverse)

#install_keras(tensorflow = 'gpu')

imdb <- dataset_imdb(num_words = 10000)

# the following two steps are identical
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb
# does the same as
train_data <- imdb$train$x
train_labels <- imdb$train$y
test_data <- imdb$test$x
test_labels <- imdb$test$y
