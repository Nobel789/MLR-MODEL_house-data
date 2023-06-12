#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 02:00:17 2023

@author: myyntiimac
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

%matplotlib inline
df=pd.read_csv("/Users/myyntiimac/Desktop/House_data.csv")
df
df.isnull().any()
df.head()
#Then check primarily any attribute that have no relevent in your dataset
#And delete this attribute
df = df.drop(['id','date'], axis = 1)
df
#After check dataframe and null check we can split the dataset into dependend and independent
#we find the price as independent and others variable are dependent and  find no catagorical column
x=df.iloc[:,1:]
x
y=df.iloc[:,0]
y
#then split the dataset for training MLR model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
#Then call the LR() from sklearn.lenearmodel and define it as MLR
from sklearn.linear_model import LinearRegression
MLR=LinearRegression()
# Then train the ddefined MLR with train data
MLR.fit(X_train,y_train)
#then check the model performance with predicting with X_test
MLR.predict(X_test)
#Then find the coefficient and intercept of MLR
MLR.coef_
MLR.intercept_
#Find the bias and variance 
bias_score=MLR.score(X_train, y_train)
bias_score
variance_score=MLR.score(X_test, y_test)
variance_score

#imports the statsmodels library instead of statemodel.formula, which provides statistical models and functions for data analysis.
import statsmodels.api as sm
# defines the function backwardElimination that takes three parameters: 
 #x (the feature matrix), y (the target variable), and SL (the significance level).
def backwardElimination(x, y, SL):
    numVars = x.shape[1]#calculates the number of variables/features in the feature matrix x using the shape attribute.
    while True:#starts an infinite loop that will continue until a break statement is encountered.
        MLR_OLS = sm.OLS(y, x).fit()#fits an ordinary least squares (OLS) regression model using sm.OLS from the statsmodels library. It fits the model using the feature matrix x and the target variable y.
        maxVar = max(MLR_OLS.pvalues)# find max p value and assign into maxVar
        if maxVar > SL:#use if conditional 
            maxVar_index = np.argmax(MLR_OLS.pvalues)#indexing with condition
            x = np.delete(x, maxVar_index, axis=1)#delete with condition
        else:
            break# maximum p-value is not greater than the significance level, the loop is terminated using the break stateme
    return x#returns the updated feature matrix x after performing backward elimination.

SL = 0.05
X_opt = x.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]].values# the initial feature matrix X_opt by selecting specific columns from X,
X_Modeled = backwardElimination(X_opt, y, SL)#call the backwardElimination function to obtain the final feature matrix X_Modeled after performing backward elimination


