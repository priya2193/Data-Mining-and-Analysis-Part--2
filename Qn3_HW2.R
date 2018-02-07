
rm(list= ls())
#Set Working Directory
#setwd("~/Desktop/SDM/Homework2/Qn2")
setwd("/Users/priyamurthy/Documents/R progamming STA545")

#library(MMST)
library(leaps)

############################################
#Generating the data set
###########################################

set.seed(1)
X <- matrix(rnorm(1000 * 20), nrow = 1000, ncol = 20)
b <- rnorm(20)
samp = round(runif(1, 1, 20))
b[samp] <- 0
samp = round(runif(1, 1, 20))
b[samp] <- 0
samp = round(runif(1, 1, 20))
b[samp] <- 0
samp = round(runif(1, 1, 20))
b[samp] <- 0
samp = round(runif(1, 1, 20))
b[samp] <- 0

E <- rnorm(1000)
Y <- X %*% b + E


############################################
#Split dataset into Training and test
###########################################

train <- sample(seq(1000), 100, replace = FALSE)
test <- -train
X.train <- X[train, ]
Y.train <- Y[train]
Y.train <- as.data.frame(Y.train)
train_data <- data.frame(X.train, Y.train)


X.test <- X[test, ]
Y.test <- Y[test]
Y.test <- as.data.frame(Y.test)
test_data <- data.frame(X.test, Y.test)

############################################
#Best Subset Selection
############################################

regfit.full <- regsubsets(Y.train~., data = train_data, nvmax = 20, method = "exhaustive", nbest = 1)
my_sum <- summary(regfit.full)

par(mfrow = c(2,2))
plot(my_sum$rss, xlab = "Number of variables", ylab = "RSS", type = "l")
plot(my_sum$adjr2, xlab = "Number of variables", ylab = "Adjusted R2", type = "l")
plot(my_sum$cp, xlab = "Number of variables", ylab = "Cp", type = "l")
plot(my_sum$bic, xlab = "Number of variables", ylab = "BIC", type = "l")
min(my_sum$cp)
which(my_sum$cp == min(my_sum$cp))
which(my_sum$bic == min(my_sum$bic))


par(mfrow = c(2,2))
plot(regfit.full, scale = "r2")
plot(regfit.full, scale = "adjr2")
plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "bic")

summary((my_sum)$outmat)



############################################
#MSE for training dataset
###########################################

err_vals = rep(NA,20)
train_mat <- model.matrix(Y.train ~., data = train_data, nvmax = 20)
for (i in 1:20){
  coefi = coef(regfit.full, id=i)
  pred = train_mat[,names(coefi)]%*%coefi
  err_vals[i] = mean((Y.train -pred)^2)
}

#quartz()
plot(err_vals, xlab = "Number of predictors", ylab = "Training MSE", pch = 19, type = "b", col = 2)

which.min(err_vals)
coef(regfit.full, which.min(err_vals))

############################################
#MSE for test dataset
###########################################

err_vals_test = rep(NA,20)
test_mat <- model.matrix(Y.test ~., data = test_data, nvmax = 20)

for (i in 1:20){
  coefi = coef(regfit.full, id=i)
  pred = test_mat[,names(coefi)]%*%coefi
  err_vals_test[i] = mean((Y.test -pred)^2)
}

quartz()
plot(err_vals_test, xlab = "Number of predictors", ylab = "Test MSE", pch = 19, type = "b", col = 2)

which.min(err_vals_test)
coef(regfit.full, which.min(err_vals_test))

###############################################################
#Compare true model with the model which has miimum test error
###############################################################

val.errors <- rep(NA, 20)
col.name = colnames(train_data)
col.name <- col.name[-c(1)]
for (i in 1:20) {
  coefi <- coef(regfit.full, id = i)
  val.errors[i] <- sqrt(sum((b[col.name %in% names(coefi)] - coefi[names(coefi) %in% col.name])^2) + sum(b[!(col.name %in% names(coefi))])^2)
}
#quartz()
plot(val.errors, xlab = "Number of coefficients", ylab = "Error between estimated and true coefficients", pch = 19, type = "b")
min_val_errors = sort(val.errors)[1]