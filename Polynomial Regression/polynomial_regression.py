#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 06:06:58 2018

@author: deathrow77
"""
# Importing Libraries
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Import dataset
data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:,1:2].values
y = data.iloc[:,2].values

# Build Linear Regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)

# Use Polynomial Features to scale the linear model to Polynomial one
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=3)
X_pr = pr.fit_transform(X)
pr.fit(X_pr, y)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_pr, y)

# Scatter Plot
plt.scatter(X, y, color='green')
plt.plot(X, lr.predict(X), color='black')
plt.plot(X, lin_reg2.predict(X_pr), color='red')

# High res plotting 
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
y_grid = lin_reg2.predict(pr.fit_transform(X_grid))
plt.plot(X_grid, y_grid, color='brown')
plt.show()

