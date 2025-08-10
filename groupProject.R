# install and load packages 
# install.packages("tidyverse")
library(tidyverse)

# set cwd
setwd(paste0("~/Library/CloudStorage/OneDrive-Personal/",
             "UofA/3.Summer2025/MIS-545/",
             "groupProject/groupProject"))
# data types
# character, factor, numeric, integer, logical, date
# load data ../scraper/used_cars_dataset_*.csv
used_cars <- read_csv(file="../scraper/used_cars_dataset_20250802_162408.csv",
                      col_types="nifffflffnllll",
                      col_names=TRUE)

print("Used Cars Dataset")
print(used_cars)
print(str(used_cars))
print(summary(used_cars))

# function for viewing all histograms
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

# call function
displayAllHistograms(used_cars)

# create derived feature of price per mile and car age
used_cars <- used_cars %>%
  mutate(price_per_mile = ifelse(mileage > 0, price / mileage, NA),
         car_age = 2025 - year)
