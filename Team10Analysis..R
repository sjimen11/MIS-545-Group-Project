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
setwd(paste0("SET_PATH_HERE"))
# Data types
# Character, factor, numeric, integer, logical, date
# Load data ../scraper/usedCarDataSet*.csv
usedCars <- read_csv(file="scraper/Team10BankingDataSet.csv.csv",
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
pkgs <- c("tidyverse", "class", "ggplot2")
to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, quiet = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))
# -------------------------------------------------------
# knn for used-cars (simple, self-contained)
# goal: classify cars as luxury (1) vs non-luxury (0)
# ----------------------------

# -----------------------------
# 2) load data + minimal prep (only if train/test not in env)

# make sure we have a binary 'isluxury' target
if ("is_luxury" %in% names(df)) {
  df <- df %>% mutate(isLuxury = as.integer(is_luxury))
} else if (!"isLuxury" %in% names(df)) {
  luxury_brands <- c("BMW","Mercedes-Benz","Audi","Lexus","Infiniti","Acura",
                     "Porsche","Jaguar","Land Rover","Genesis","Cadillac","Volvo")
  df <- df %>% mutate(isLuxury = as.integer(make %in% luxury_brands))
} else {
  df <- df %>% mutate(isLuxury = as.integer(isLuxury))
}

# keep rows with the basics we need
df <- df %>% tidyr::drop_na(price, mileage, year, make, isLuxury)

# simple 75/25 split
set.seed(545)
idx <- sample(seq_len(nrow(df)), size = round(0.75 * nrow(df)))
usedCarTraining <- df[idx, , drop = FALSE]
usedCarTesting  <- df[-idx, , drop = FALSE]


# -------------------------------------------------------
# 3) prepare matrices for knn

# target should be factor (0/1)
usedCarTraining$isLuxury <- factor(usedCarTraining$isLuxury, levels = c(0,1))
usedCarTesting$isLuxury  <- factor(usedCarTesting$isLuxury,  levels = c(0,1))

# predictors = all columns except target
x_cols <- setdiff(names(usedCarTraining), "isLuxury")
fml <- as.formula(paste("~", paste(x_cols, collapse = " + "), "-1"))

x_train <- model.matrix(fml, data = usedCarTraining)
x_test  <- model.matrix(fml, data = usedCarTesting)

# make test columns match train before scaling
missing_in_test <- setdiff(colnames(x_train), colnames(x_test))
if (length(missing_in_test) > 0) {
  addm <- matrix(0, nrow = nrow(x_test), ncol = length(missing_in_test))
  colnames(addm) <- missing_in_test
  x_test <- cbind(x_test, addm)
}
extra_in_test <- setdiff(colnames(x_test), colnames(x_train))
if (length(extra_in_test) > 0) {
  x_test <- x_test[, setdiff(colnames(x_test), extra_in_test), drop = FALSE]
}
x_test <- x_test[, colnames(x_train), drop = FALSE]
stopifnot(identical(colnames(x_train), colnames(x_test)))

# scale using training stats (so every feature counts fairly)
centers <- colMeans(x_train)
scales  <- apply(x_train, 2, sd); scales[scales == 0] <- 1
x_train_sc <- scale(x_train, center = centers, scale = scales)
x_test_sc  <- scale(x_test,  center = centers, scale = scales)

y_train <- usedCarTraining$isLuxury
y_test  <- usedCarTesting$isLuxury

# ----------------------
# 4) run knn
k_value <- 5
knn_pred <- class::knn(train = x_train_sc, test = x_test_sc, cl = y_train, k = k_value)

# results
knn_cm <- table(actual = y_test, predicted = knn_pred)
print(knn_cm)

knn_acc <- mean(knn_pred == y_test)
cat("knn (k =", k_value, ") accuracy:", round(knn_acc, 4), "\n")

# ----------------------
# 5) two quick visuals for slides

# (a) k vs accuracy
k_values <- seq(1, 15, 2)
acc_values <- sapply(k_values, function(k) {
  p <- class::knn(train = x_train_sc, test = x_test_sc, cl = y_train, k = k)
  mean(p == y_test)
})
k_df <- data.frame(k = k_values, accuracy = acc_values)

p_k <- ggplot(k_df, aes(x = k, y = accuracy)) +
  geom_line() + geom_point() +
  geom_vline(xintercept = k_value, linetype = "dashed", color = "red") +
  labs(title = "k vs accuracy",
       subtitle = paste("chosen k =", k_value),
       x = "neighbors (k)", y = "accuracy") +
  theme_minimal()

print(p_k)
ggplot2::ggsave("knn_k_vs_accuracy.png", p_k, width = 7, height = 4.5, dpi = 300)

# (b) confusion matrix heatmap
cm_df <- as.data.frame(knn_cm); colnames(cm_df) <- c("actual","predicted","count")

p_cm <- ggplot(cm_df, aes(x = predicted, y = actual, fill = count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = count), fontface = "bold") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = paste("confusion matrix (k =", k_value, ")"),
       x = "predicted", y = "actual", fill = "count") +
  theme_minimal()

print(p_cm)
ggplot2::ggsave("knn_confusion_heatmap.png", p_cm, width = 7, height = 4.5, dpi = 300)

cat("saved: knn_k_vs_accuracy.png, knn_confusion_heatmap.png\n")

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


