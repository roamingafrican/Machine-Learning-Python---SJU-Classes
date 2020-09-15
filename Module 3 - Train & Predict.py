#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 11:51:35 2019

@author: jan-willemkleynhans
"""

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import math
import random
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

#Import dataset
from sklearn.datasets.california_housing import fetch_california_housing

caliHousing = fetch_california_housing()
df = caliHousing

#Describing dataset
print(caliHousing.DESCR)
print(caliHousing.feature_names)
print(caliHousing.data.shape)
print(caliHousing.data[100])
print(math.exp(caliHousing.target[0]))

#Samples
samples = caliHousing.data.shape[0]
print(samples)

#Data split into training_, validation_and test_sets
#training_set = 70% of the dataset
#validation_set = 15% of the dataset
#test_set = 15% of the dataset
##########
slice_points = [0, 0.7, 0.85, 1]
slice_points = list(zip(slice_points, slice_points[1:]))
print(slice_points)
slice_points = np.array(slice_points) * samples
print(slice_points)
train_inputs, valid_inputs, tests_inputs = [caliHousing.data[int(start):int(stop)] for start, stop in slice_points]
train_inputs.shape, valid_inputs.shape, tests_inputs.shape
train_labels, valid_labels, tests_labels = [caliHousing.target[int(start): int(stop)] for start, stop in slice_points]
assert train_inputs.shape[0] == train_labels.shape[0]
assert valid_inputs.shape[0] == valid_labels.shape[0]
assert tests_inputs.shape[0] == tests_labels.shape[0]
######

model = LinearRegression()
model.fit(X = train_inputs, y = train_labels)
model.score(train_inputs, train_labels)
LinearRegression(copy_X = True, fit_intercept = True, n_jobs = 1, normalize = False)
valid_pred = model.predict(valid_inputs)
train_pred = model.predict(train_inputs)

print(valid_pred)
print(train_pred)
s = [1, 10, 100]
print(valid_labels, valid_pred)
print((np.array([ 4.5, 5.00001, 4.259]), np.array([ 2.41614771, 2.50028634, 3.33485518])))
print(mse(train_labels, model.predict(train_inputs)))
print(mse(valid_labels, valid_pred))

plt.scatter(train_labels[train_labels < 4], train_pred[train_labels < 4])
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices")

int(math.exp(0.8) * 1000)
np.mean(train_inputs[:, 0]), np.std(train_inputs[:, 0])
new_features = StandardScaler().fit_transform(X=train_inputs)
feature = 1
round(np.mean(new_features[:, 1]), 5), np.std(new_features[:, 1])
model_1 = Pipeline([
    ('normalizer', StandardScaler()),
    ('regressor', LinearRegression())
]).fit(train_inputs, train_labels)
def loss(model):
    train_loss = mse(train_labels, model.predict(train_inputs))
    valid_loss = mse(valid_labels, model.predict(valid_inputs))
    return train_loss, valid_loss

loss(model_1)
model_2 = Pipeline([
    ("normalizer", StandardScaler()),
    ("poli-feature", PolynomialFeatures(degree=2)),
    ("regressor", LinearRegression())
])
model_2.fit(train_inputs, train_labels)
Pipeline(steps=[('normalizer', StandardScaler(copy=True, with_mean=True, with_std=True)), ('poli-feature', PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)), ('regressor', LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False))])
s = [1, 10, 100]
valid_labels[s], model_2.predict(valid_inputs[s])
(np.array([ 4.5    ,  5.00001,  4.259  ]), np.array([ 2.95001462,  2.96427055,  3.51599654]))
valid_pred = model_2.predict(valid_inputs)
print(mse(valid_labels, valid_pred))

def chart(labels, predictions):
    plt.scatter(labels, predictions)
    plt.xlabel("Prices: $Y_i$")
    plt.ylabel("Predicted prices: $\hat{Y}_i$")
    plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
chart(valid_labels, model_2.predict(valid_inputs))

print(sum(valid_pred < 0))

mse(valid_labels[valid_pred > 0], valid_pred[valid_pred > 0])
math.exp(0.7) * 1000
chart(valid_labels[valid_pred > 0], valid_pred[valid_pred > 0])
print(loss(model_2))
permutation = np.random.permutation(samples)
caliHousing.data[permutation] = caliHousing.data
caliHousing.target[permutation] = caliHousing.target
train_inputs, valid_inputs, tests_inputs = [caliHousing.data[int(start):int(stop)] for start, stop in slice_points]
train_labels, valid_labels, tests_labels = [caliHousing.target[int(start): int(stop)] for start, stop in slice_points]
model_2.fit(train_inputs, train_labels)
print(loss(model_2))
math.exp(0.38) * 1000

model_3 = Pipeline([
    ('normalizer', StandardScaler()),
    ('poly-feat', PolynomialFeatures(degree=3)),
    ('regressor', Ridge())
]).fit(train_inputs, train_labels)

mse(tests_labels, model_1.predict(tests_inputs))
mse(tests_labels, model_2.predict(tests_inputs))
mse(tests_labels, model_3.predict(tests_inputs))
print(mse(tests_labels, model_1.predict(tests_inputs)))
print(mse(tests_labels, model_2.predict(tests_inputs)))
print(mse(tests_labels, model_3.predict(tests_inputs)))