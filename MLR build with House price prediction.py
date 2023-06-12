#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 23:55:35 2023

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
#these scores indicate that the MLR model performs reasonably well in explaining and predicting the variation in the dependent variable based on the given independent variables. 
#However, it is important to note that there may still be some unexplained variation or noise in the data that is not captured by the model.
#now we are going to find which variable have highly corelate with dependent variable price
#we try to find the which independent variable have highly statistical relation with dependent variable
#this done by tecnique called Backward elimination where we delete less relevent independent variable with the value of satatistical p_value
#We set treshhold p_value is less than 0.05, if any attribute p_value  greater than 0.05 , we eliminate
#before that We add our model derived intercept value into df then do backward elemination with OLS()
x = np.append(arr=np.full((21613, 1), 3684411), values=x, axis=1) 
x
##Then do the recursive feature elemination with OLS
#For this, we import statsmodel.formula as sm and from sm we use OLS
#And define  alist for independent variable 
import statsmodels.api as sm
X_opt = x[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18]]


#OrdinaryLeastSquares
MLS_OLS = sm.OLS(endog=y, exog=X_opt).fit()

MLS_OLS.summary()
# we find the variable X5 have higher p_value which is greater than 0.05
#Lets find the OLS score
import statsmodels.api as sm
X_opt = x[:, [0, 1, 2, 3, 4,6,7,8,9,10,11,12,13,14,15,16,17,18]]


#OrdinaryLeastSquares
MLS_OLS = sm.OLS(endog=y, exog=X_opt).fit()

MLS_OLS.summary()
##From OLS summary we can see that all  attribute  have high relation with price except floor

