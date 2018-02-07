
rm(list= ls())
#Set Working Directory
setwd("~/Desktop/SDM/Homework2/Qn2")

train_data <- read.table("ticdata2000.txt")
test_data <- read.table("ticeval2000.txt")
target_data <- read.table("tictgts2000.txt")
colnames(target_data)[colnames(target_data) == 'V1'] <- 'V86'
testfull <- cbind(test_data, target_data)
set.seed(1)


#################################################################
### Bar Graph for purchase of caravan policy vs Customer Subtype
#################################################################
a<-table(Caravan$MOSTYPE[Caravan$Purchase=="Yes"])
barplot(a, main="Purchase of Caravan policy vs Customer subtype" ,xlab="Customer Subtype", ylab="Number of customers")

########################################################################
### Bar Graph for purchase of caravan policy vs purchase of boat policy
########################################################################
a<-table(Caravan$APLEZIER[Caravan$Purchase=="Yes"])
barplot(a, main="Purchase of Caravan policy vs Purchase of boat policy" ,xlab="Number of boat policies", ylab="Number of customers")



#######################################
#Linear Model
#######################################
result<-lm(train_data$V86~.,data=train_data)
print(summary(result))
#Predict and find error
pred <- predict(result, test_data)
lm_error <-  mean((target_data - pred)^2)




#######################################
#Forward Selection
#######################################
set.seed(123)
regfit.fwd <- regsubsets(train_data$V86~., data = train_data, nvmax = 85, method = "forward", nbest = 1)
my_sum <- summary(regfit.fwd)
summary((my_sum)$outmat)
#error
err_vals_test = rep(NA,85)
test_mat <- model.matrix(testfull$V86 ~., data = testfull, nvmax = 85)
for (i in 1:85){
  coefi = coef(regfit.fwd, id=i)
  pred = test_mat[,names(coefi)]%*%coefi
  err_vals_test[i] = mean((testfull$V86 -pred)^2)
}
which.min(err_vals_test)
min.err_vals_test <- sort(err_vals_test)[1]
print(min.err_vals_test)
coef(regfit.fwd, which.min(err_vals_test))



#######################################
#Backward Selection
#######################################

set.seed(123)
regfit.bwd <- regsubsets(train_data$V86~., data = train_data, nvmax = 85, method = "backward")
my_sum <- summary(regfit.bwd)
summary((my_sum)$outmat)
#error
err_vals_test_bwd = rep(NA,85)
test_mat <- model.matrix(testfull$V86 ~., data = testfull, nvmax = 85)
for (i in 1:85){
  coefi = coef(regfit.bwd, id=i)
  pred_bwd = test_mat[,names(coefi)]%*%coefi
  err_vals_test_bwd[i] = mean((testfull$V86 -pred_bwd)^2)
}
which.min(err_vals_test_bwd)
min.err_vals_test_bwd <- sort(err_vals_test_bwd)[1]
print(min.err_vals_test_bwd)
coef(regfit.bwd, which.min(err_vals_test_bwd))

#######################################
#Ridge Regresssion
#######################################
set.seed(12345)
colnames(target_data)[colnames(target_data) == 'V1'] <- 'V86'
test_data_wr <- cbind(test_data, target_data)
train.mat = model.matrix(train_data$V86~., data=train_data)
test.mat = model.matrix(test_data_wr$V86~., data=test_data_wr)
grid = 10 ^ seq(4, -2, length=100)
ridge.mod = glmnet(train.mat,train_data$V86, alpha = 1e-15, lambda = grid)
ridge.mod.cv = cv.glmnet(train.mat,train_data$V86, alpha = 1e-15, lambda = grid, nfolds = 50)
lambda.best = ridge.mod.cv$lambda.min

ridge.pred <- predict(ridge.mod, s = lambda.best, type ="coefficients")
ridge.pred2 <- predict(ridge.mod, s = lambda.best, newx = test.mat, type ="response")

##Calculating error:
y_hat <- ridge.pred2
y_true <- test_data_wr$V86
test_error <- sum((y_hat - y_true)^2)

mean((y_hat - y_true)^2)



#######################################
#Lasso Regresssion
#######################################

lasso.mod <- glmnet(train.mat,train_data$V86, alpha = 1)
lasso.mod.cv = cv.glmnet(train.mat,train_data$V86, alpha = 1, nfolds = 50)
lambda.best.lasso = lasso.mod.cv$lambda.min


set.seed(12345)

lasso.pred <- predict(lasso.mod, s = lambda.best.lasso, type ="coefficients")
lasso.pred2 <- predict(lasso.mod, s = lambda.best.lasso, newx = test.mat, type ="response")


y_hat_lasso <- lasso.pred2
y_true <- test_data_wr$V86
test_error_lasso <- sum((y_hat_lasso - y_true)^2)

mean((y_hat_lasso - y_true)^2)



