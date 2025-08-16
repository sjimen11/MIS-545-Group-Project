# Install and load packages 
# install.packages("tidyverse")
# install.packages("rpart")
# install.packages("rpart.plot")

# load required libraries
library(tidyverse)
library(rpart)
library(rpart.plot)


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

# Noramalization
usedCars2 <- usedCars %>%
  mutate(logPrice = log(price),
         logMileage = log(mileage))

displayAllHistograms(usedCars2)

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

# K-nearest neighbors

# Naive Bayes

# Decision tree
# Use only useful predictors
usedCarDecisionTreeModel <- rpart(highPriced ~ year + make + mileage ,
                                  data = usedCarTesting,
                                  method = "class",
                                  cp = 0.05)

# Plot the decision tree model
rpart.plot(usedCarDecisionTreeModel)

# Predict FarmOwnership for the testing data
usedCarPredictions <- predict(usedCarDecisionTreeModel,
                               usedCarTesting,
                               type = "class")
# Display riceFarmPredictions
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

