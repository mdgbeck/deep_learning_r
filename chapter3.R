library(keras)

#install_keras(tensorflow = 'gpu')

imdb <- dataset_imdb(num_words = 10000)

# the following two steps are identical
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

# # does the same as
# train_data <- imdb$train$x
# train_labels <- imdb$train$y
# test_data <- imdb$test$x
# test_labels <- imdb$test$y

# data is endcoding of sequence of words limited to top 10000 most common words
train_data[[1]]
max(sapply(train_data, max))

# translate integers into english words
word_index <- dataset_imdb_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index

decode_review <- function(index){
  if (index >= 3) word <- reverse_word_index[[as.character(index - 3)]]
  else word <- "?"
}

decoded_review <- purrr::map_chr(train_data[[1]], decode_review)
paste(decoded_review, collapse = " ")


# need to turn into tensor do that by changing to binary matrix
# how to manually do it by hand
# returns matrix where each row is a review and each column indicates whether
# a word was included in the review, losses number of words and order of words
vectorize_sequences <- function(sequences, dimension = 10000){
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for(i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

# create partial training set
val_indices <- 1:10000

x_val <- x_train[val_indices, ]
partial_x_train <- x_train[-val_indices, ]

y_val <- matrix(y_train[val_indices], ncol = 1)
partial_y_train <- matrix(y_train[-val_indices], ncol = 1)


# model
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation  = "sigmoid")

# compile the models (show a few different options)
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 4,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

str(history)
plot(history)

model %>% predict(x_test[1:10, ]) %>% round(2)

cat(map_chr(test_data[[7]], decode_review))



# section 3.5 multi classification
library(keras)
library(tidyverse)

reuters <- dataset_reuters(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% reuters

vectorize_sequences <- function(sequences, dimension = 10000){
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for(i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}
x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

one_hot_train_labels <- to_categorical(train_labels)
one_hot_test_labels <- to_categorical(test_labels)


model <- keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 46, activation = "softmax")

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

val_indicies <- 1:1000

x_val <- x_train[val_indicies, ]
partial_x_train <- x_train[-val_indicies, ]

y_val <- one_hot_train_labels[val_indicies, ]
partial_y_train <- one_hot_train_labels[-val_indicies, ]

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 9,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)
plot(history)

results <- model %>% evaluate(x_test, one_hot_test_labels)
results

prediction <- model %>% predict(x_test)
dim(prediction)
sum(prediction[1, ])
round(prediction[1, ], 2)



# section 3.6 regression example
library(keras)
library(tidyverse)

dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset

data_mean <- apply(train_data, 2, mean)
data_sd <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = data_mean, scale = data_sd)
test_data <- scale(test_data, center = data_mean, scale = data_sd)

build_model <- function(){
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu",
                input_shape = dim(train_data)[[2]]) %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 1)
  
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae")
  )
}

# k fold validation
k <- 4
indices <- sample(1:nrow(train_data))
folds <- cut(1:length(indices), breaks = k, labels = FALSE)

num_epochs <- 15
all_mae_histories <- NULL
all_scores <- c()
for(i in 1:k){
  cat("processing fold #", i, "\n")
  
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- train_data[val_indices, ]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices, ]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model()

  history <- model %>% fit(
    partial_train_data,
    partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = num_epochs,
    batch_size = 1
  )
  
  mae_history <- history$metrics$val_mean_absolute_error
  all_mae_histories <- rbind(all_mae_histories, mae_history)

  results <- model %>% evaluate(val_data, val_targets, verbose = 0)
  all_scores <- c(all_scores, results$mean_absolute_error)
  
}

average_mae_history <- data_frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories, 2, mean)
)

ggplot(average_mae_history, aes(epoch, validation_mae)) +
  geom_line() +
  geom_smooth()


# train final model
model <- build_model()
model %>% fit(train_data, 
              train_targets,
              epochs = 80,
              batch_size = 16
          )
result <- model %>% evaluate(test_data, test_targets)
result
