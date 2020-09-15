
# coding: utf-8

# In[23]:


#LIBRARIES
import mglearn as mglearn
import os
import graphviz
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
import sklearn.svm as linear_svm
from mpl_toolkits.mplot3d import Axes3D, axes3d
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import make_circles
from sklearn.datasets import load_wine
from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPRegressor


# In[2]:


#DECISION TREE
wine = load_wine()
print(wine.DESCR)
df = pd.DataFrame(wine.data,columns=wine.feature_names)
df['target'] = pd.Series(wine.target)
df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
              'Alcalinity of ash', 'Magnesium', 'Total phenols',
              'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
              'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
X = df.drop('Class label', 1)
y = df['Class label']
df.head()


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(wine.data, 
                                                    wine.target, 
                                                    stratify = wine.target, 
                                                    random_state = 42)
tree = DecisionTreeClassifier(criterion = 'entropy',
                             min_samples_split = 20,
                             min_samples_leaf = 15,
                             max_features = 'sqrt',
                             max_leaf_nodes = 12,
                             random_state = 0)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print('#Training data points: %d' % X_train.shape[0])
print('#Testing data points: %d' % X_test.shape[0])
print("")
print("Accuracy on training set: {:,.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:,.3f}".format(tree.score(X_test, y_test)))
print("")
print('Class labels:', np.unique(wine.target))
print('Misclassified samples: %d' % (y_test != y_pred).sum())


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(wine.data, 
                                                    wine.target, 
                                                    stratify = wine.target, 
                                                    random_state = 42)

tree = DecisionTreeClassifier(criterion = 'entropy',
                             min_samples_split = 20,
                             min_samples_leaf = 15,
                             max_features = 'sqrt',
                             max_leaf_nodes = 12,
                             random_state = 0)
tree.fit(X_train, y_train)

export_graphviz(tree, out_file = "tree.dot",
                feature_names = wine.feature_names,
                filled = True, rounded = True,
               special_characters = True) #[writes data into .dot file]

with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))

print("Feature importance: \n{}".format(tree.feature_importances_))

def plot_feature_importances_wine(model):
    n_features = wine.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), wine.feature_names)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
plot_feature_importances_wine(tree)

tree = mglearn.plots.plot_tree_not_monotone()

display(tree)


# In[15]:


#RANDOM FOREST
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators = 50, 
                                 min_samples_split = 20,
                                min_samples_leaf = 15,
                                max_features = 'sqrt',
                                max_leaf_nodes = 12,
                                random_state = 0)
forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))


importances = forest.feature_importances_
features = wine['feature_names']
indices = np.argsort(importances)



plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()



# In[6]:


estimator = forest.estimators_[29]

export_graphviz(estimator, out_file = "rftree.dot", 
                feature_names = wine.feature_names,
                class_names = wine.target_names,
                rounded = True, proportion = False,
                precision = 2, filled = True)


with open("rftree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


# In[16]:


importances = forest.feature_importances_

#sorted indices
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            X.columns.values[indices[f]], 
                            importances[indices[f]]))


# In[21]:


#Gradient Boosted Trees MaxDepth = 0.01
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=0)

gbrt=GradientBoostingClassifier(random_state=0, learning_rate=0.01)

gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

importances = gbrt.feature_importances_
features = wine['feature_names']
indices = np.argsort(importances)

def plot_feature_importances_wine(model):
       n_features = wine.data.shape[1] 
       plt.barh(range(n_features), model.feature_importances_, align='center')
       plt.yticks(np.arange(n_features), wine.feature_names)
       plt.xlabel("Feature importance")
       plt.ylabel("Fetaure")
       plt.ylim(-1, n_features)   
       plot_feature_importances_wine(gbrt)


# In[9]:


#Gradient Boosted Trees MaxDepth = 0.05
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=0)

gbrt=GradientBoostingClassifier(random_state=42, learning_rate=0.05)

gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

importances = gbrt.feature_importances_
features = wine['feature_names']
indices = np.argsort(importances)

def plot_feature_importances_wine(model):
       n_features = wine.data.shape[1] 
       plt.barh(range(n_features), model.feature_importances_, align='center')
       plt.yticks(np.arange(n_features), wine.feature_names)
       plt.xlabel("Feature importance")
       plt.ylabel("Fetaure")
       plt.ylim(-1, n_features)   
       plot_feature_importances_wine(gbrt)


# In[10]:


#Gradient Boosted Trees MaxDepth = 0.1
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=0)

gbrt=GradientBoostingClassifier(random_state=42, learning_rate=0.1)

gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

importances = gbrt.feature_importances_
features = wine['feature_names']
indices = np.argsort(importances)

def plot_feature_importances_wine(model):
       n_features = wine.data.shape[1] 
       plt.barh(range(n_features), model.feature_importances_, align='center')
       plt.yticks(np.arange(n_features), wine.feature_names)
       plt.xlabel("Feature importance")
       plt.ylabel("Fetaure")
       plt.ylim(-1, n_features)   
       plot_feature_importances_wine(gbrt)


# In[11]:


#Gradient Boosted Trees MaxDepth = 1
X_train, X_test, y_train, y_test = train_test_split(wine.data, 
                                                    wine.target, 
                                                    stratify = wine.target, 
                                                    random_state = 0)

gbrt=GradientBoostingClassifier(random_state=42, max_depth=1)

gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

importances = gbrt.feature_importances_
features = wine['feature_names']
indices = np.argsort(importances)

def plot_feature_importances_wine(model):
       n_features = wine.data.shape[1] 
       plt.barh(range(n_features), model.feature_importances_, align='center')
       plt.yticks(np.arange(n_features), wine.feature_names)
       plt.xlabel("Feature importance")
       plt.ylabel("Fetaure")
       plt.ylim(-1, n_features)   
       plot_feature_importances_wine(gbrt)


# In[24]:


ch = fetch_california_housing()
df = pd.DataFrame(ch.data,columns=ch.feature_names)
df.head()


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(ch.data, ch.target, random_state=42)

mlp = MLPRegressor(hidden_layer_sizes = (20,),
                  activation = 'relu',
                  solver = 'adam',
                  learning_rate = 'adaptive',
                  max_iter = 2000,
                  learning_rate_init = 0.01,
                  alpha = 0.01,
                  random_state = 0).fit(X_train, y_train)


# In[27]:


print("Adjusted R-square on training set: {:.3f}".format(mlp.score(X_train, y_train)))
print("Adjusted R-square on test set: {:.3f}".format(mlp.score(X_test, y_test)))

