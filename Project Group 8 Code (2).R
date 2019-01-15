# Please Import the Train Dataset to run the code
#Code for (Data1):
library(dplyr)
data = train_set0506
data = subset(data, select = -c(Cat2, Cat4, Cat5, Cat7))
head(data)
data$C_Claim = ifelse(data$Claim_Amount > 0, 1, 0)
table(data$C_Claim)
data[data == '?'] <- NA
subset(data, is.na(OrdCat))
table(data$OrdCat, data$C_Claim)
#Data Imputation
library(Hmisc)
table(data$OrdCat)
data$OrdCat <- impute(data$OrdCat, median)
#NA Removals
Cat1[Cat1 == "?"] <- NA
Cat3[Cat3 == "?"] <- NA
Cat6[Cat6 == "?"] <- NA
Cat8[Cat8 == "?"] <- NA
Cat10[Cat10 == "?"] <- NA
Cat11[Cat11 == "?"] <- NA
Cat12[Cat12 == "?"] <- NA
Blind_Make[Blind_Make == "?"] <- NA
#Conversion to Factors
data$Cat1 <- as.factor(data$Cat1)
data$Cat3 <- as.factor(data$Cat3)
data$Cat6 <- as.factor(data$Cat6)
data$Cat8 <- as.factor(data$Cat8)
data$Cat9 <- as.factor(data$Cat9)
data$Cat10 <- as.factor(data$Cat10)
data$Cat11 <- as.factor(data$Cat11)
data$Cat12 <- as.factor(data$Cat12)
data$Blind_Make <- as.factor(data$Blind_Make)
data$Blind_Model <- as.factor(data$Blind_Submodel)
data$Blind_Submodel <- as.factor(data$Blind_Submodel)
data$NVCat <- as.factor(data$NVCat)
data$Model_Year <- as.factor(data$Model_Year)
#Count total number of 1's removed after data filtering
a <- c(which(is.na(data$Cat1)), which(is.na(data$Cat3)),
       which(is.na(data$Cat6)), which(is.na(data$Cat8)),
       which(is.na(data$Cat10)), which(is.na(data$Cat11)),
       which(is.na(data$Cat12)), which(is.na(Blind_Make)))
sum(data$C_Claim[a])
data_1 <- na.omit(data)
table(data_1$C_Claim)
summary(data_1)
#Subset Data
library(mlr)
data_new = subset(data_1, select = -c(X1, Household_ID, Calendar_Year,
                                      Blind_Model, Blind_Submodel, Claim_Amount))
data_new <- createDummyFeatures(data_new)
#Sample Data in Positive and Negative
library(dplyr)
data_pos <- subset(data_new, C_Claim == 1)
data_neg <- subset(data_new, C_Claim == 0)
#Make Test and Train Sets
data_pos_trainsamp <- sample(1:nrow(data_pos), 0.7*nrow(data_pos))
data_neg_trainsamp <- sample(1:nrow(data_neg), 0.7*nrow(data_neg))
data_train_pos <- data_pos[data_pos_trainsamp, ]
data_train_neg <- data_neg[data_neg_trainsamp, ]
data_train <- rbind(data_train_pos, data_train_neg)
data_test_pos <- data_pos[-data_pos_trainsamp, ]
data_test_neg <- data_neg[-data_neg_trainsamp, ]
data_test <- rbind(data_test_pos, data_test_neg)
sampos <- sample(nrow(data_train_pos), 500)
samneg <- sample(nrow(data_train_neg), 600)
data_sampos <- data_pos[sampos,]
data_samneg <- data_neg[samneg,]
data_samp <- rbind(data_sampos, data_samneg)
#Logistic Regression on Sample Data
model <- glm(C_Claim ~., data_samp, family = binomial)
summary(model)
#Prediction on Sample Data
prob <- predict(model, data_samp, type = "response")
glm.pred <- rep(0, length(prob))
glm.pred[prob > 0.5] <- 1
table(glm.pred, data_samp$C_Claim)
hist(prob)
library(pROC)
roc_logit <- roc(C_Claim ~ prob, data = data_samp)
plot(roc_logit)
auc(roc_logit)
#Prediction on Test Data
prob <- predict(model, data_test, type = "response")
glm.pred <- rep(0, length(prob))
glm.pred[prob > 0.5] <- 1
table(glm.pred, data_test$C_Claim)
hist(prob)
library(pROC)
roc_logit <- roc(C_Claim ~ prob, data = data_test)
plot(roc_logit)
auc(roc_logit)
#SVM
library(e1071)
colnames(data_samp)
svm_fit <- svm(factor(C_Claim)~.-NVCat.D-Cat12.A, data_samp, kernel =
                 "radial", cost = 100, gamma = 0.01)
pred <- predict(svm_fit, data_test, decision.values = TRUE)
table(pred, data_test$C_Claim)
#Adaptive Boosting
library(adabag)
train_samp_ada <- data_samp
test_samp_ada <- data_test
table(test_samp_ada$C_Claim)
train_samp_ada$C_Claim <- as.factor(train_samp_ada$C_Claim)
data.adaboost <- boosting(C_Claim ~., train_samp_ada, coeflearn="Freund",
                          mfinal = 100, control = (minsplit = 0))
data.adaboost.pred <- predict.boosting(data.adaboost, test_samp_ada)
data.adaboost.pred$confusion