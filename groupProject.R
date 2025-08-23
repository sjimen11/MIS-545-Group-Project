# Install and load packages 
# install.packages("tidyverse")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("class")
# install.packages("corrplot")
# install.packages("olsrr")
# install.packages("ggcorrplot")
# install.packages("e1071") 
# install.packages("caret")

# load required libraries
library(tidyverse)
library(rpart)
library(rpart.plot)
library(class)
library(corrplot)
library(olsrr)
library(ggcorrplot)
library(e1071)
library(caret)
library(ggplot2)

# Set cwd
setwd(paste0("~/Library/CloudStorage/OneDrive-Personal/",
             "UofA/3.Summer2025/MIS-545/",
             "groupProject/groupProject"))
# Data types
# Character, factor, numeric, integer, logical, date
# Load data ../scraper/usedCarDataSet*.csv
usedCars <- read_csv(file="scraper/usedCarDataSet20250811_201429.csv",
                      col_types="nifffflffnllll",
                      col_names=TRUE)
# Display results
print("Used Cars Dataset")
print(usedCars)
str(usedCars)
summary(usedCars)

# Function for viewing all histograms
# Define displayAllHistograms() function
displayAllHistograms <- function(tibbleDataset) {
  tibbleDataset %>%
    keep(is.numeric) %>%
    gather() %>%
    ggplot() + 
    geom_histogram(mapping = aes(x = value, fill = key), color = "black") +
    facet_wrap(~ key, scales = "free") +
    theme_minimal()
}

# Create derived feature of price per mile and car age
usedCars <- usedCars %>%
  mutate(pricePerMile = ifelse(mileage > 0, price / mileage, NA),
         carAge = 2025 - year)

# Call function
displayAllHistograms(usedCars)

# Remove any missing rows with missing values in price_per_mile or car_age
usedCars <- usedCars %>%
  filter(!is.na(pricePerMile) & !is.na(carAge))

# Correlation plot of numeric variables
numericVars <- usedCars %>%
  select(price, mileage, carAge, pricePerMile)

corMatrix <- cor(numericVars, use = "complete.obs")

ggcorrplot(corMatrix, 
           hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           method="circle", 
           colors = c("red", "white", "blue"), 
           title="Correlation Plot of Numeric Variables", 
           ggtheme=theme_minimal())

# Noramalization
usedCars2 <- usedCars %>%
  mutate(logPrice = log(price),
         logMileage = log(mileage))

displayAllHistograms(usedCars2)

# Scatterplot with log-transformed variables
ggplot(usedCars2, aes(x = logMileage, y = logPrice, color = isLuxury)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", color = "black") +
  theme_minimal() +
  labs(title = "Scatterplot: Log(Price) vs Log(Mileage) by Luxury",
       x = "Log(Mileage)",
       y = "Log(Price)")

# Boxplot: Price by Luxury
ggplot(usedCars2, aes(x = as.factor(isLuxury), y = price, fill = as.factor(isLuxury))) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::dollar) +
  theme_minimal() +
  labs(title = "Boxplot: Price by Luxury vs Non-Luxury",
       x = "Is Luxury",
       y = "Price")

# Queries
# Average price by vehicle make
avgPriceByMake <- usedCars %>%
  group_by(make) %>%
  summarize(avgPrice = mean(price, na.rm = TRUE)) %>%
  arrange(desc(avgPrice))

# Display results
print("Average Price by Vehicle Make")
print(avgPriceByMake)
str(avgPriceByMake)
summary(avgPriceByMake)

# Top 10 makes by average price
AvgPriceByMake_Top10 <- avgPriceByMake %>%
  slice_max(order_by = avgPrice, n = 10)

ggplot(AvgPriceByMake_Top10,
       aes(x = fct_reorder(make, avgPrice), y = avgPrice)) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(labels = scales::dollar) +
  labs(title = "Average Price by Make (Top 10)",
       x = "Make", y = "Average Price ($)") +
  theme_minimal()

# Average mileage by vehicle make
avgMileageByMake <- usedCars %>%
  group_by(make) %>%
  summarize(avgMileage = mean(mileage, na.rm = TRUE)) %>%
  arrange(desc(avgMileage))

# Display results
print("Average Mileage by Vehicle Make")
print(avgMileageByMake)
str(avgMileageByMake)
summary(avgMileageByMake)

# Top 10 makes by average mileage
AvgMileageByMake_Top10 <- avgMileageByMake %>%
  slice_max(order_by = avgMileage, n = 10)

ggplot(AvgMileageByMake_Top10,
       aes(x = fct_reorder(make, avgMileage), y = avgMileage)) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Average Mileage by Make (Top 10)",
       x = "Make", y = "Average Mileage") +
  theme_minimal()

# Average price per mile by vehicle make
avgPricePerMileByMake <- usedCars %>%
  group_by(make) %>%
  summarise(avgPricePerMile = mean(pricePerMile, na.rm = TRUE)) %>%
  arrange(desc(avgPricePerMile))

# Display results
print("Average Mileage by Vehicle Make")
print(avgPricePerMileByMake)
str(avgPricePerMileByMake)
summary(avgPricePerMileByMake)

# Top 10 makes by average price per mile
AvgPricePerMileByMake_Top10 <- avgPricePerMileByMake %>%
  slice_max(order_by = avgPricePerMile, n = 10)

ggplot(AvgPricePerMileByMake_Top10,
       aes(x = fct_reorder(make, avgPricePerMile), y = avgPricePerMile)) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(labels = function(x) scales::dollar(x)) +
  labs(title = "Average Price per Mile by Make (Top 10)",
       x = "Make", y = "Average Price per Mile ($/mile)") +
  theme_minimal()

# Set seed for splitting data 
randomSeed <- sample(1:999, 1)
set.seed(randomSeed)

# Create sample set
sampleSet <- sample(nrow(usedCars2),
                    round(nrow(usedCars2) * 0.75),
                    replace = FALSE)

# Split data into 75% training data
usedCarTraining <- usedCars2[sampleSet, ]

# Split data into 25% testing data
usedCarTesting <- usedCars2[-sampleSet, ]

# Logistic regression
# Generating Logistic regression model
usedCarsLRModel <- glm(data = usedCarTraining , 
                     family = binomial , 
                     formula = highPriced ~ year + mileage + 
                       isLuxury + isRecent)
# Displaying the model
summary(usedCarsLRModel)

# Predict classes for each record in the testing dataset
usedCarsLRPrediction <- predict(usedCarsLRModel,
                            usedCarTesting,
                          type = "response"
)

# Display the predictions from usedCarsprediction on console
print(usedCarsLRPrediction)

# Converting to 0 & 1 probabilities
usedCarsLRClass <- ifelse(usedCarsLRPrediction > 0.5, 1, 0)

# Evaluate the model by forming a confusion matrix
usedCarsLRConfusionMatrix <- table(usedCarTesting$highPriced,
                                 usedCarsLRPrediction)

# Display the confusion matrix on the console
print(usedCarsLRConfusionMatrix)

# Calculate the model predicitive accuracy
LRpredictiveAccuracy <- sum(diag(usedCarsLRConfusionMatrix)) /
  nrow(usedCarTesting)

# print the predictiveAccuracy
print(LRpredictiveAccuracy)

# K-nearest neighbors
# ------------------------------
# k-nearest neighbors (knn model)
# goal: predict if a car is luxury (1) or not luxury (0)
# based on features like price, mileage, year, make, etc.
# ------------------------------

# make sure target is isLuxury (factor with 0/1 instead of number)
usedCarTraining$isLuxury <- factor(usedCarTraining$isLuxury, levels = c(0,1))
usedCarTesting$isLuxury  <- factor(usedCarTesting$isLuxury,  levels = c(0,1))

# take all columns except isLuxury as predictors
# turn categorical variables (like make) into dummy columns (0/1)
# build predictor matrices for training and testing
x_cols <- setdiff(names(usedCarTraining), "isLuxury")
fml <- as.formula(paste("~", paste(x_cols, collapse = " + "), "-1"))
x_train <- model.matrix(fml, data = usedCarTraining)
x_test  <- model.matrix(fml, data = usedCarTesting)

# scale numeric features so no single feature (like price) dominates
# puts everything on the same scale
centers <- colMeans(x_train)
scales  <- apply(x_train, 2, sd)
scales[scales == 0] <- 1
x_train_sc <- scale(x_train, center = centers, scale = scales)
x_test_sc  <- scale(x_test,  center = centers, scale = scales)

# set up target values
y_train <- usedCarTraining$isLuxury
y_test  <- usedCarTesting$isLuxury

# pick a k value (here just 5 neighbors for simplicity)
# knn predicts luxury based on the closest 5 cars
k_value <- 5

# run knn on the test data
knn_pred <- knn(train = x_train_sc, test = x_test_sc, cl = y_train, k = k_value)

# confusion matrix shows actual vs predicted results
knn_cm <- table(actual = y_test, predicted = knn_pred)
print(knn_cm)

# accuracy = percent of cars classified correctly
knn_acc <- mean(knn_pred == y_test)
cat("knn (k =", k_value, ") accuracy:", knn_acc, "\n")

# Naive Bayes
# Convert target to factor (classification) 
usedCars$highPriced <- as.factor(usedCars$highPriced) 

# Select relevant predictors 
carData3 <- usedCars %>% select(highPriced, year, mileage, make, isLuxury, isRecent) 

# Split into training (75%) and testing (25%)  
trainIndex <- createDataPartition(carData3$highPriced, p = 0.75, list = FALSE) 

carTrain <- carData3[trainIndex, ] 
carTest <- carData3[-trainIndex, ] 

# Train Naive Bayes model 
carNB <- naiveBayes(highPriced ~ year + mileage + make + isLuxury + isRecent, 
                    data = carTrain) 

# View model details 
print(carNB) 

# Predict on test set 
carPred <- predict(carNB, carTest) 

# Confusion matrix for accuracy
confusionMatrix(carPred, carTest$highPriced)

# Decision tree
# Use only useful predictors
usedCarDecisionTreeModel <- rpart(highPriced ~ year + make + mileage ,
                                  data = usedCarTesting,
                                  method = "class",
                                  cp = 0.05)

# Plot the decision tree model
rpart.plot(usedCarDecisionTreeModel)

# Predict highPriced for the testing data
usedCarPredictions <- predict(usedCarDecisionTreeModel,
                               usedCarTesting,
                               type = "class")
# Display usedCarPredictions
print(usedCarPredictions)

# Create a confusion matrix
usedCarConfusionMatrix <- table(usedCarTesting$isLuxury,
                                usedCarPredictions)
# Display the confusion matrix
print(usedCarConfusionMatrix)

# Calculate the accuracy of the model
predictiveAccuracy <- sum(diag(usedCarConfusionMatrix)) /
  nrow(usedCarTesting)

# Display the predictive accuracy
print(predictiveAccuracy)


