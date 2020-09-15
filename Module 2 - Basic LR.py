
# coding: utf-8

# In[72]:


import numpy as np
from sklearn.model_selection import train_test_split
import random

X, y = np.arange(40).reshape((10, 4)), range(10)
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


# In[73]:


print('X_train')
print(X_train)
print('X_test')
print(X_test)

np.random.shuffle(X_train)
X_train[:,0]


# In[91]:


from random import shuffle
np.random.shuffle(X_train)
X_train[:]


# In[92]:


np.random.shuffle(X_test)
X_test[:]


# In[95]:


np.random.shuffle(y_test)
y_test[:]



# In[97]:


np.random.shuffle(y_train)
y_train[:]


# In[98]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn


# In[99]:


from sklearn.datasets import load_boston


# In[110]:


boston = load_boston()
print(boston.keys())


# In[117]:


boston.DESCR


# In[116]:


boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()


# In[126]:


boston['MEDV'] = boston_dataset.target
boston.isnull().sum()


# In[138]:


from sklearn.model_selection import train_test_split
Y = boston['MEDV']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[149]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()


plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')


# In[160]:


from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']


# In[161]:


lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[162]:


y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("                                      ")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("                                      ")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

