
rm(list= ls())

#Set Working Directory
setwd("/Users/priyamurthy/Documents/R progamming STA545")

library(pls)
library(ISLR)
library(glmnet)
data(College)


write.table(College, file= "College_data.txt", sep = "\t", col.names = names(College))
data_set <- read.delim("College_data.txt", sep = "\t", header = TRUE)



###############################################
#Split the dataset intro Training and Test set
###############################################

#dt = sort(sample(nrow(data_set), nrow(data_set)*.7))
#train = sample(nrow(data_set), nrow(data_set)*.7)
train = sample(1:nrow(data_set), round(nrow(data_set)/2))
test<- -train

train_data <- College[train, ]
test_data <- College[test, ]

##################################################
#Fit a linear model using least squares on the 
#training set, and report the test error obtained
#################################################
result<-lm(train_data$Apps~.,data=train_data)
print(summary(result))

#Predict and find error
pred <- predict(result, test_data)
lm_error <-  mean((test_data[, "Apps"] - pred)^2)


######################################################################
#FFit a ridge regression model on the training set, 
#with λ chosen by cross- validation. Report the test error obtained.
######################################################################
set.seed(12345)
train.mat = model.matrix(train_data$Apps~., data=train_data)
test.mat = model.matrix(test_data$Apps~., data=test_data)
grid = 10 ^ seq(4, -2, length=100)
ridge.mod = glmnet(train.mat,train_data[, "Apps"], alpha = 1e-15, lambda = grid)
ridge.mod.cv = cv.glmnet(train.mat,train_data[, "Apps"], alpha = 1e-15, lambda = grid, nfolds = 50)

#Look at different lambdas

ridge.mod$lambda[100]
coef(ridge.mod)[,100]
l2_norm <- sqrt(sum(coef(ridge.mod)[2:19, 100]^2))
lambda.best = ridge.mod.cv$lambda.min


#model selection

set.seed(12345)
ridge.pred <- predict(ridge.mod, s = lambda.best, type ="coefficients")
ridge.pred2 <- predict(ridge.mod, s = lambda.best, newx = test.mat, type ="response")

##Calculating error:
y_hat <- ridge.pred2
y_true <- test_data[, "Apps"]
test_error_ridge <- mean(((y_hat - y_true)^2))


###################################################################
# 1.c Fit a lasso model on the training set, 
#with λ chosen by crossvalidation. Report the test error 
#obtained, along with the number of non-zero coefficient estimates.
###################################################################


lasso.mod <- glmnet(train.mat,train_data[, "Apps"], alpha = 1)
lasso.mod.cv = cv.glmnet(train.mat,train_data[, "Apps"], alpha = 1, nfolds = 50)
lambda.best.lasso = lasso.mod.cv$lambda.min
print(lambda.best.lasso)

coef(lasso.mod, s =lambda.best.lasso)

set.seed(12345)
lasso.pred <- predict(lasso.mod, s = lambda.best.lasso, type ="coefficients")
lasso.pred2 <- predict(lasso.mod, s = lambda.best.lasso, newx = test.mat, type ="response")

#Find errors
y_hat_lasso <- lasso.pred2
y_true <- test_data[, "Apps"]
test_error_lasso <- mean((y_hat_lasso - y_true)^2)



####################################################################################
#1.e Fit a PCR model on the training set, with k chosen by cross-validation. 
#Report the test error obtained, along with the value of k selected by cross-validation.
####################################################################################
set.seed(2)
pcr.fit = pcr(train_data$Apps ~. , data = train_data, scale = TRUE, validation = "CV")
summary(pcr.fit)
quartz()
validationplot(pcr.fit, val.type = "MSEP",  main = "Apps")


# Evaluate performance of the model with "i" components in the pcr regression for test.
set.seed(3)
test_error_store <- c()
y_true <- test_data[, "Apps"]
dim(test_data)
for (i in 1:17){
  pcr.pred.test = predict(pcr.fit, test_data, ncomp = i)
  test.error <- mean((pcr.pred.test-y_true)^2)
  test_error_store <- c(test_error_store, test.error)
}

#quartz()
plot(test_error_store)
test_error_pcr <- sort(test_error_store)[1]

#########################################
#Fit a PLS model on the training set, 
#with k chosen by crossvalidation.
#Report the test error obtained, along with the value of k selected by cross-validation.
#########################################

pls.fit = plsr(train_data$Apps ~., data = train_data, scale = TRUE, validation = "CV")
summary(pls.fit)
quartz()
validationplot(pls.fit, val.type = "MSEP", main = "Apps")


set.seed(3)
#training_error_store <- c()
pls_test_error_store <- c()
y_true <- test_data[, "Apps"]
for (i in 1:17){
  pls.pred.test = predict(pls.fit, test_data, ncomp = i)
  pls.test.error <- mean((pls.pred.test-y_true)^2)
  pls_test_error_store <- c(test_error_store, pls.test.error)
}

quartz()
plot(pls_test_error_store)

test_error_pls <- sort(pls_test_error_store)[1]


#########################################
#Comparing all
#########################################

test_avg <- mean(test_data$Apps)
lm_r2 <- 1 - lm_error / mean((test_avg - test_data$Apps)^2)
ridge_r2 <- 1 - test_error_ridge / mean((test_avg - test_data$Apps)^2)
lasso_r2 <- 1 - test_error_lasso / mean((test_avg - test_data$Apps)^2)
pcr_r2 <- 1 - test_error_pcr / mean((test_avg - test_data$Apps)^2)
pls_r2 <- 1 - test_error_pls / mean((test_avg - test_data$Apps)^2)
