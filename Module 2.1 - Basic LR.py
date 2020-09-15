#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:44:20 2019

@author: jan-willemkleynhans
"""

import pandas as pd
import numpy as np
import mglearn as mglearn
import sklearn
from scipy import stats
from sklearn.model_selection import train_test_split

import os
print(os)

X, y = np.arange(10).reshape((5, 2)), range(5)
print("X:\n{}".format(X))
print("y:\n{}".format(list(y)))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("X_train:\n{}".format(X_train))
print("y_train:\n{}".format(y_train))
print("X_test:\n{}".format(X_test))
print("y_test:\n{}".format(y_test))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle = False)
print("y_train:\n{}".format(y_train))
print("y_test:\n{}".format(y_test))

from sklearn.datasets import load_iris
iris_dataset = load_iris()


print("Keys of iris_dataset:  \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data:  {}".format(iris_dataset['data'].shape))
print("First five rows of data:\n{}".format(iris_dataset['data'][:5]))
print("Type of target: {}".format(type(iris_dataset['target'])))
print("Shape of target:  {}".format(iris_dataset['target'].shape))
print("Target:\n{}".format(iris_dataset['target']))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape: {}".format(X_train.shape))
print("y-train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y-test shape: {}".format(y_test.shape))


iris_dataframe =  pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)


cadataset = pd.read_csv("housing.csv")
cadataset.describe()
cadataset.hist('total_rooms')
import matplotlib.pyplot as plt
X = cadataset[['total_rooms']].values
y = cadataset[['median_house_value']].values
plt.scatter(X, y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
from sklearn import linear_model
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
#This results in creating a linear fit.  We can then make a prediction using the test set
y_pred = regressor.predict(X_test)
#Create the linear fit using the train set
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.xlabel('total_rooms')
plt.ylabel('median_house_value')
plt.show()
#Apply that linear fit to the test set
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Total Rooms vs. Median House Value (Test Set)')
plt.xlabel('total_rooms')
plt.ylabel('median_house_value')
plt.show()