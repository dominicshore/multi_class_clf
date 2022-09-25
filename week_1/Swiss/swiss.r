library(tidyverse)
library(caret)

# Load the data
data("swiss")
# Inspect the data
sample_n(swiss, 3)


# Split the data into training and test set
set.seed(123)
training.samples <- swiss$Fertility %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- swiss[training.samples, ]
test.data <- swiss[-training.samples, ]
# Build the model
model <- lm(Fertility ~., data = train.data)
# Make predictions and compute the R2, RMSE and MAE
predictions <- model %>% predict(test.data)
data.frame( R2 = R2(predictions, test.data$Fertility),
            RMSE = RMSE(predictions, test.data$Fertility),
            MAE = MAE(predictions, test.data$Fertility)) 
            # MAE is Mean Absolute Error