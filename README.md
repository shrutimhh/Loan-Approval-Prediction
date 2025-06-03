# Loan-Approval-Prediction

## Project Proposal Code## 
#load in the dataset
loandata <- read.csv(file.choose())
loandata

#changing categorical to binary (Graduate = 1, Not Graduate = 0)
loandata$education<- ifelse(loandata$education == " Graduate",1,0)

#Categorical to binary (Yes = 1, No = 0)
loandata$self_employed<- ifelse(loandata$self_employed == " Yes",1,0)

#Categorical to binary (Approved = 1, Rejected = 0)
loandata$loan_status<- ifelse(loandata$loan_status == " Approved",1,0)

summary(loandata)

boxplot(loandata[,2:4]) #dependents, education, self employed
boxplot(loandata[,5:6]) #income, loan amount
boxplot(loandata[,7]) #loan term
boxplot(loandata[,8]) #cibil score
boxplot(loandata[,9:12]) #res assets, comm assets, lux assets, bank asset
boxplot(loandata[,13]) #loan status

model_1 <- glm(loan_status ~ ., family = binomial, data = loandata)
model_1

## CODE

#load in the dataset from kaggle
# (download dataset from kaggle and choose file when screen pops up)
loandata <- read.csv(file.choose())
loandata

loandata$loan_status<- as.factor(loandata$loan_status)
loandata$loan_status<- as.numeric(loandata$loan_status)
loandata <- as.factor(loandata)

install.packages("randomForest")
library(randomForest)
install.packages('ROCR')
library(ROCR)


library(randomForest)
library(datasets)
library(caret)

summary(loandata)
dev.cur()
dev.off()
dev.new()
install.packages("ggplot2")
boxplot(loandata[,2:4]) #dependents, education, self employed
boxplot(loandata[,5:6]) #income, loan amount
boxplot(loandata[,7]) #loan term
boxplot(loandata[,8]) #cibil score
boxplot(loandata[,9:12]) #res assets, comm assets, lux assets, bank asset
boxplot(loandata[,13]) #loan status


index <- sample(nrow(loandata),nrow(loandata)*0.90) #for training and testing set
loan_train = loandata[index,]
loan_test = loandata[-index,]






#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~RANDOM FORESTS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#training a random forest model with different ntree values
ntree_values <- c(50,75,100,125,150,175,200,225,250,275,300) 
oob_errors <- numeric(length(ntree_values))

#for OOB rate of each value
for (i in 1:length(ntree_values)) {
  rf_model <- randomForest(loan_status ~ .-loan_id, data = loan_train, ntree = ntree_values[i])
  oob_errors[i] <- rf_model$err.rate[nrow(rf_model$err.rate), "OOB"]}

#plot of OOB error vs ntree
plot(ntree_values, oob_errors, type = "b", 
     xlab = "Number of Trees (ntree)", 
     ylab = "Out-of-Bag Error Rate",
     main = "OOB Error Rate vs. ntree")
# 200 is the best value to use bc it has the lowest oob error rate

#random forest using 200 trees and mtry of 3 bc theres 11 variables so sqrt(11) is approx. 3
rf <- randomForest(loan_status ~ . - loan_id, data = loan_train, 
                   ntree = 200, mtry = 3, oob.prox = TRUE)
print(rf)
plot(rf)

#confusion matrix for training set 
p1 <- predict(rf, loan_train)
confusionMatrix(p1, loan_train$loan_status)
# 100% accuracy obviously bc this is the set this was trained on

#confusion matrix for test set
p2 <- predict(rf, loan_test)
confusionMatrix(p2, loan_test$loan_status)
# 98.59% accuracy which is very good


#ROC curve & AUC
rf_ROC <- predict(rf, type="prob",newdata = loan_test)[,2]
rf_ROC1 <- prediction(rf_ROC, loan_test$loan_status)
plot(performance(rf_ROC1, "tpr", "fpr"), lwd = 2, colorize=TRUE)
plot(rf_ROC1)
rf_AUC1 <- performance(rf_ROC1, measure = "auc")@y.values[[1]] 
rf_AUC1


#for variable importance
varImpPlot(rf,
           sort = T,
           n.var = 10,
           main = "Top 10 - Variable Importance")
importance(rf)
#we can see that cibil score is most important by a lot






#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LOGISTIC REGRESSION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#this is for logistic regression
#changing categorical to binary (Graduate = 1, Not Graduate = 0)
loandata$education<- ifelse(loandata$education == " Graduate",1,0)

#Categorical to binary (Yes = 1, No = 0)
loandata$self_employed<- ifelse(loandata$self_employed == " Yes",1,0)

#Categorical to binary (Approved = 1, Rejected = 0)
loandata$loan_status<- ifelse(loandata$loan_status == " Approved",1,0)


model_1 <- glm(loan_status ~ . - loan_id , family = binomial, data = loan_train)
model_1 # all except loan id since its useless

#to find best model (forward)
nullmodel = glm(loan_status ~ 1, data = loandata)
fullmodel = glm(loan_status ~ ., data = loandata)
model.step <- step(nullmodel, scope = list(lower = nullmodel,
                                           upper = fullmodel),
                   direction = "forward")
model_opt_train <- glm(loan_status ~ cibil_score + loan_term + loan_amount + income_annum, 
                       family = binomial, data = loan_train)
model_opt_test <- glm(loan_status ~ cibil_score + loan_term + loan_amount + income_annum, 
                      family = binomial, data = loan_test)

AIC(model_1)
AIC(model_opt_train) #this ones better

#for in sample prediction and misclass table using confusion matrix
#using cutoff value of 0.5
pred_loan_train <- predict(model_opt_train, type = "response")
table(loan_train$loan_status, (pred_loan_train > 0.5)*1, dnn=c("Truth","Predicted"))

#for out of sample prediction and misclass table with 0.5 cutoff value again
pred_loan_test <- predict(model_opt_test, type = "response")
table(loan_test$loan_status, (pred_loan_test > 0.5)*1, dnn=c("Truth","Predicted"))


#ROC curve & AUC
pred0 <- prediction(pred_loan_test, loan_test$loan_status)
perf0 <- performance(pred0, "tpr", "fpr")
plot(perf0, colorize=TRUE) #ROC curve for test set
slot(performance(pred0,"auc"), "y.values")[[1]]







#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CLASSIFICATION TREE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#classification tree
install.packages('rpart')
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)

loan_rpart0 <- rpart(formula = loan_status ~ ., data = loan_train, method =
                       "class")
loan_rpart <- rpart(formula = loan_status ~ . , data = loan_train, method =
                      "class",parms = list(loss=matrix(c(0,5,1,0), nrow = 2)))
loan_rpart
pred0 <- predict(loan_rpart0, type="class")
table(loan_train$loan_status, pred0, dnn = c("True", "Pred"))
prp(loan_rpart, extra = 1)
#this shows that cibil score is most important

#in sample
loan_train.pred.tree1<- predict(loan_rpart, loan_train, type="class")
table(loan_train$loan_status, loan_train.pred.tree1, dnn=c("Truth","Predicted"))

#out of sample
loan_test.pred.tree1<- predict(loan_rpart, loan_test, type="class")
table(loan_test$loan_status, loan_test.pred.tree1, dnn=c("Truth","Predicted"))

#does this part return 0 for you guys?
cost <- function(r, phat){
  weight1 <- 5
  weight0 <- 1
  pcut <- weight0/(weight1+weight0)
  c1 <- (r==1)&(phat<pcut) #logical vector - true if actual 1 but predict 0
  c0 <-(r==0)&(phat>pcut) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0))
}
cost(loan_train$loan_status, predict(loan_rpart, loan_train, type="prob"))




#For ROC curve in sample
#in sample
#predicted prob on train set
loan_train_prob_rpart = predict(loan_rpart, loan_train, type="prob")
pred = prediction(loan_train_prob_rpart[,2], loan_train$loan_status)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#for AUC
slot(performance(pred,"auc"), "y.values")[[1]]



#For ROC curve out of sample
#out of sample
#predicted prob on test set
loan_test_prob_rpart = predict(loan_rpart, loan_test, type="prob")
predC = prediction(loan_test_prob_rpart[,2], loan_test$loan_status)
perfC = performance(predC, "tpr", "fpr")
plot(perfC, colorize=TRUE)
#for AUC
slot(performance(predC,"auc"), "y.values")[[1]]

