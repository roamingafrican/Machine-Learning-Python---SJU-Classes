
# coding: utf-8

# In[53]:


#                                                LIBRARIES
import numpy as np
import mglearn as mglearn
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.pipeline import Pipeline


# In[22]:


print("Algorithm Chains")


# In[23]:


cancer = load_breast_cancer()
scaler = MinMaxScaler().fit(X_train)
svm = SVC()


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 66)


# In[15]:


X_train_scaled = scaler.transform(X_train)
svm.fit(X_train_scaled, y_train)
X_test_scaled = scaler.transform(X_test)


# In[18]:


print("Test score: {:.2f}".format(svm.score(X_test_scaled, y_test)))


# In[25]:


print("Parameter Selections with preprocessing")


# In[28]:


param_grid = {'C':[0.001, 0.01, 0.1, 1, 10, 100],
             'gamma':[0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(SVC(), param_grid = param_grid, cv = 5)
grid.fit(X_train_scaled, y_train)


# In[29]:


print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)
print("Test set accuracy: {:.2f}".format(grid.score(X_test_scaled, y_test)))


# In[36]:


mglearn.plots.plot_improper_processing()


# In[37]:


print("Building Pipelines")


# In[34]:


pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
pipe.fit(X_train, y_train)


# In[35]:


print("Test score:: {:.2f}".format(pipe.score(X_test, y_test)))


# In[41]:


print("Using pipeline in GridSearch")


# In[44]:


param_grid = {'svm__C':[0.001, 0.01, 0.1, 1, 10, 100],
             'svm__gamma':[0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train, y_train)


# In[45]:


print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
print("Best parameters: {}".format(grid.best_params_))


# In[61]:


mglearn.plots.plot_proper_processing()


# In[62]:


print("Illustrate infromation leakage")


# In[58]:


rnd = np.random.RandomState(seed = 0)
X = rnd.normal(size = (100, 10000))
y = rnd.normal(size = (100, ))


# In[59]:


select = SelectPercentile(score_func = f_regression, percentile = 5).fit(X, y)
X_selected = select.transform(X)
print("X_selected.shape: {}".format(X_selected.shape))


# In[60]:


print("Cross-validation accuracy (cv only on ridge): {:.2f}".format(np.mean(cross_val_score(Ridge(), 
                                                                                            X_selected, y, cv= 5))))


# In[63]:


pipe = Pipeline([("select", SelectPercentile(score_func = f_regression, percentile = 5)), ("ridge", Ridge())])
print("Cross validation accuracy: {:.2f}".format(np.mean(cross_val_score(pipe, X, y, cv = 5))))

