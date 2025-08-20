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
