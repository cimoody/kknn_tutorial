# Tutorial from
# https://predictoanalycto.wordpress.com/2014/06/05
# /classification-using-k-nearest-neighbour-knn-algorithm-in-r-programming-languagepart-1/
#
# followed by CIMoody 8/15

# Install and load packages
install.packages("kknn");
library(kknn);

# Load the data from iris
data(iris);
# View the data
fix(iris);
# Partitioning data into training and testing sets
m <- nrow(iris);
imp <- sample(1:m, m/3, prob = rep(1/m, m));
iris.train <- iris[-imp, ];
iris.test <- iris[imp, ];
# Setting up and creating model to predict Species
iris.knn <- kknn(formula = formula(Species ~ . ), train = iris.train, test = iris.test, k = 7, distance = 1);
# Extracting the prediction
fit <- fitted(iris.knn);
table(iris.test$Species, fit);


# tutorial from
# https://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Classification/kNN
# followed by CIMoody 8/15

# Install and load packages
install.packages("RWeka", dependencies = TRUE);
library(RWeka);
# Load and read data into RWeka format
iris <- read.arff(system.file("arff", "iris.arff", package = "RWeka"));
# Simple example choosing k
classifier <- IBk(class ~., data = iris);
summary(classifier);
#
# === Summary ===
#
#     Correctly Classified Instances         150              100      %
# Incorrectly Classified Instances         0                0      %
# Kappa statistic                          1
# Mean absolute error                      0.0085
# Root mean squared error                  0.0091
# Relative absolute error                  1.9219 %
# Root relative squared error              1.9335 %
# Coverage of cases (0.95 level)         100      %
# Mean rel. region size (0.95 level)      33.3333 %
# Total Number of Instances              150
#
# === Confusion Matrix ===
#
#     a  b  c   <-- classified as
# 50  0  0 |  a = Iris-setosa
# 0 50  0 |  b = Iris-versicolor
# 0  0 50 |  c = Iris-virginica

# Another example letting RWeka find the best value for k
classifier <- IBk(class ~ ., data = iris, control = Weka_control(K = 20, X = TRUE));
evaluate_Weka_classifier(classifier, numFolds = 10);
# === 10 Fold Cross Validation ===
#
#     === Summary ===
#
#         Correctly Classified Instances         144               96      %
#     Incorrectly Classified Instances         6                4      %
#     Kappa statistic                          0.94
#     Mean absolute error                      0.0457
#     Root mean squared error                  0.1389
#     Relative absolute error                 10.2933 %
#     Root relative squared error             29.4696 %
#     Coverage of cases (0.95 level)         100      %
#     Mean rel. region size (0.95 level)      42      %
#     Total Number of Instances              150
#
#     === Confusion Matrix ===
#
#         a  b  c   <-- classified as
#     50  0  0 |  a = Iris-setosa
#     0 48  2 |  b = Iris-versicolor
#     0  4 46 |  c = Iris-virginica
classifier;
# IB1 instance-based classifier
# using 6 nearest neighbour(s) for classification
