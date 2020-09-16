
# coding: utf-8

# In[193]:


#                                               LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn as mglearn
import graphviz
import pprint

from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder
from IPython.display import Image
from sklearn import datasets
from mglearn.datasets import make_blobs
from sklearn.svm import SVC

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


from sklearn.metrics import classification_report, confusion_matrix


# In[20]:


pp = pprint.PrettyPrinter(indent=4)
wine = load_wine()
logreg = LogisticRegression()


# In[45]:


#Structuring wine dataset
print(wine.DESCR)
wine = datasets.load_wine()
df = pd.DataFrame(wine.data,columns=wine.feature_names)
print(df)


# In[56]:


df = pd.DataFrame(wine.data,columns=wine.feature_names)
df['target'] = pd.Series(wine.target)
df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
              'Alcalinity of ash', 'Magnesium', 'Total phenols',
              'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
              'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
X = df.drop('Class label', 1)
y = df['Class label']
df.head()
print("The shape of features: ", df.shape)
print()
print(df.describe())


# In[60]:


#One-hot encode
df = pd.get_dummies(df)
df.iloc[:,5:].head(5)


# In[88]:


#splitting dataset
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target,  
                                                    test_size = .20,random_state = 42)
print("Split Shape")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print()
feature_list = list(df.columns)


# In[79]:


#Linear Regression Model
print("==Linear Regression model accuracy scores==")
lm = LinearRegression()
model = lm.fit(X_train, y_train)
predictions = model.predict(X_test)


# In[6]:


#Plotting Actual vs Predicted
plt.scatter(y_test, predictions)
plt.xlabel("Actual")
plt.ylabel("Predicted")


# In[91]:


#Linear Model training and testing stats
print('# of Training data points for (LM): %d' % X_train.shape[0])
print('# of Testing data points for (LM): %d' % X_test.shape[0])
print()
print("Accuracy on training set (for LM): {:,.3f}".format(lm.score(X_train, y_train)))
print("Accuracy on test set (for LM): {:,.3f}".format(lm.score(X_test, y_test)))
print()
print('Class labels:', np.unique(wine.target))
print('Misclassified samples: %d' % (y_test != predictions).sum())
print()
errors = abs(predictions - y_test)
print("Mean absolute error:{:.3f}".format(np.mean(errors)))


# In[166]:


#Building Random Forest
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators = 50, 
                                min_samples_split = 20,
                                min_samples_leaf = 15,
                                max_features = 'sqrt',
                                max_leaf_nodes = 12,
                                random_state = 0)
forest.fit(X_train, y_train)


# In[167]:


#Display Random Forest
estimator = forest.estimators_[1]
export_graphviz(estimator, out_file = "rftree.dot", 
                feature_names = wine.feature_names,
                class_names = wine.target_names,
                rounded = True, proportion = False,
                precision = 2, filled = True)


with open("rftree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


# In[114]:


#Random Forest results
print("Feature importance (Random Forest):\n{}".format(forest.feature_importances_))
print()

y_pred = forest.predict(X_test)
print("Accuracy on training set (Random Forest): {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set (Random Forest): {:.3f}".format(forest.score(X_test, y_test)))


# In[138]:


#Organizing feature importances for Random Forest
importances = forest.feature_importances_
#sorted indices
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            X.columns.values[indices[f]], 
                            importances[indices[f]]))


# In[139]:


print()
features = wine['feature_names']
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[169]:


cumulative_importances = np.cumsum(importances)
print('Number of features for 95% importance:', np.where(cumulative_importances > 0.95)[0][0]+1)
#Cross Validation
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size = .20, random_state = 0)
forest_cv_score = cross_val_score(forest, wine.data, wine.target, cv=10)


# In[189]:


rf = RandomForestClassifier()
pp.pprint(rf.get_params())
print()

n_estimators = [int(x) for x in np.linspace(start = 20, stop = 40, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pp.pprint(random_grid)
print()

rf_random = RandomizedSearchCV(estimator = rf,
                       param_distributions = random_grid,
                       n_iter = 10,
                       cv = 3,
                       verbose = 2,
                       random_state = 42,
                       n_jobs = -1)
rf_random.fit(X_train, y_train)


# In[190]:


rf_random.best_params_


# In[191]:


predictions = rf_random.predict(X_test)
print("Accuracy on training set (RF2): {:,.3f}".format(rf_random.score(X_train, y_train)))
print("Accuracy on test set (RF2): {:,.3f}".format(rf_random.score(X_test, y_test)))
print()
print('Class labels:', np.unique(wine.target))
print('Misclassified samples: %d' % (y_test != predictions).sum())
print()
errors = abs(predictions - y_test)
print("Mean absolute error:{:.3f}".format(np.mean(errors)))


# In[195]:


#Create grid based on random search
param_grid = {
'bootstrap': [True],
'max_depth': [80, 90, 100, 110],
'max_features': [2, 3],
'min_samples_leaf': [3, 4, 5],
'min_samples_split': [8, 10, 12],
'n_estimators': [100, 200, 300, 1000]
}

rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf,
                           param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)


# In[196]:


grid_search.best_params_


# In[148]:


print("=====CONFUSION MATRIX=====")
print(confusion_matrix(y_test, forest.predict(X_test)))
print('\n')

print("=====CLASSIFICATION REPORT=====")
print(classification_report(y_test, forest.predict(X_test)))
print('\n')

print("AUC SCORES")
print(forest_cv_score)
print('\n')
print("MEAN AUC SCORES")
print("Mean AUC Score - Random Forest: ", forest_cv_score.mean())


# In[ ]:


#


# In[199]:


#


# In[201]:


#grid_search= GridSearchCV(SVC(), grid, cv = 10, return_train_score = True)
#print(grid_search)


# In[202]:


scores = cross_val_score(logreg, wine.data, wine.target)
print("Cross-validation scores:{}".format(scores))
print()
scores = cross_val_score(logreg, wine.data, wine.target, cv = 5)
print("Cross-validation scores:{}".format(scores))
print()
print("Average cross-validation score:{:.2f}".format(scores.mean()))


# In[203]:


kfold = KFold(n_splits = 5)
scores = cross_val_score(logreg, wine.data, wine.target, cv = kfold)
print()
print("Cross-validation scores:\n{}".format(scores))
kfold = KFold(n_splits = 3)
scores = cross_val_score(logreg, wine.data, wine.target, cv = kfold)
print()
print("Cross-validation scores:\n{}".format(scores))
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
scores = cross_val_score(logreg, wine.data, wine.target, cv=kfold)
print()
print("Cross-validation scores: \n{}".format(scores))


# In[204]:


loo = LeaveOneOut()
scores = cross_val_score(logreg, wine.data, wine.target, cv=loo)
print("Number of cv iterations: ", len(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))


# In[205]:


print("Wine labels:\n{}".format(wine.target))
mglearn.plots.plot_stratified_cross_validation()


# In[207]:


#split data into train & validation & test
X_trainval, X_test, y_trainval, y_test = train_test_split(wine.data, wine.target, 
                                                    random_state=0)

#split train & validation into train & validation
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, 
                                                    random_state=1)

print("Size of training set: {}  size of validation set: {}  size of tets set:"
      "{}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
best_score=0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        #for each combination of parameters, train an SVC
        svm = SVC(gamma=gamma, C=C)
        #perform cross-validation
        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
        #compute mean cross-validation accuracy
        score = np.mean(scores)
        # if get better score, store score and parameters
        if score > best_score:
            best_score = score
            best_parameters = {'C':C, 'gamma':gamma}
            
# rebuild a model on the combined training and validation set,
# and evalaute it on the test set
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
mglearn.plots.plot_cross_val_selection()


# In[209]:


rf=RandomForestClassifier()
rf.fit(X_train, y_train)
rf.fit(X_test, y_test)
rf.fit(X_trainval, y_trainval)


# In[212]:


print("Accuracy on training set (Final RF): {:,.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on training set (Final RF): {:,.3f}".format(rf.score(X_trainval, y_trainval)))
print("Accuracy on test set (Final RF): {:,.3f}".format(rf.score(X_test, y_test)))

