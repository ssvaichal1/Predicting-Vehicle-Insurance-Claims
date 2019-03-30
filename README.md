# Predicting-Vehicle-Insurance-Claims

This dataset was provided as part of the capstone project for ISEN-613 (Engineering Data Analysis) course. In a dataset containing 100000
entries, only 720 were insurance claims. As this is a highly unbalanced dataset, oversampling techniques such as SMOTE were used to prepare
the training data. MICE library in R was used to impute missing values. As the data contained both numeric and categorical data, dummy
variables were created. 

Logistic regression, SVM and Adaptive boosting were found to be the best models for predicting insurance claims based on 
vehicle owner and vehicle information data. 
