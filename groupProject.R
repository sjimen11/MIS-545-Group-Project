# Install and load packages 
# install.packages("tidyverse")
library("tidyverse")

# Set cwd
setwd(paste0("SET_PATH_HERE"))
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

# Logistic regression

# K-nearest neighbors

# Naive Bayes

# Decision tree