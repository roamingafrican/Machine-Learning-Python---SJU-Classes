
# coding: utf-8

# In[1]:


#                                           Library Imports
import mglearn as mglearn
import os
import sklearn.datasets as datasets
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image  
from sklearn.tree import export_graphviz
import graphviz
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
import sklearn.svm as linear_svm
from mpl_toolkits.mplot3d import Axes3D, axes3d
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
from sklearn.datasets import make_circles


# In[2]:


#Activity
#[Using IRIS dataset]
iris=datasets.load_iris()
mglearn.plots.plot_animal_tree()
df=pd.DataFrame(iris.data, columns=iris.feature_names)   #[Creating the dataframe]
y=iris.target
dtree=DecisionTreeClassifier()
dtree.fit(df,y)
export_graphviz(dtree, out_file="tree_iris.dot",  
                filled=True, rounded=True,
                special_characters=True)
with open("tree_iris.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


# In[3]:


#[Creating unpruned Decision Tree w/ cancer dataset]
cancer = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree=DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:,.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

#[Creating pruned Decision Tree w/ cancer dataset]
cancer = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state = 42)
tree = DecisionTreeClassifier(max_depth = 4, random_state = 0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:,.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:,.3f}".format(tree.score(X_test, y_test)))


# In[4]:


#[Analyzing the Decision Trees]
export_graphviz(tree, out_file = "tree.dot", impurity = False, filled = True) #[writes data into .dot file]
with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))

print("Feature importance: \n{}".format(tree.feature_importances_))

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
plot_feature_importances_cancer(tree)

tree = mglearn.plots.plot_tree_not_monotone()

display(tree)


# In[5]:


#[Use dataset of historical comp memory prices]
ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))
plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel("Year")
plt.ylabel("Price in $/Mbyte")


                                                  #[use historical data to forecast prices after the year 2000]
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

X_train = data_train.date[:, np.newaxis]    #[predict prices based on date]

y_train = np.log(data_train.price)   #[we use a log-transform to get a simpler relationship of data to target]
tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

X_all = ram_prices.date[:, np.newaxis]    #[predict on all data]
pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

price_tree = np.exp(pred_tree)  #[undo log-transform]
price_lr = np.exp(pred_lr)
plt.semilogy(data_train.date, data_train.price, label="Training data")
plt.semilogy(data_test.date, data_test.price, label="Test data")
plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
plt.legend()


# In[6]:


#[Analyzing random forest]
X, y = make_moons(n_samples = 100, noise = 0.25, random_state = 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42)
forest = RandomForestClassifier(n_estimators = 5, random_state = 2)
forest.fit(X_train, y_train)
fig, axes = plt.subplots(2, 3, figsize = (20,10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax = ax)
mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)
axes[-1,-1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train)


# In[7]:


#[Random Forest with Breast Cancer Dataset]
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
forest=RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
importances = forest.feature_importances_
features = cancer['feature_names']
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[8]:


#[Gradient Boosted Trees (Breast Cancer Dataset - Learning Rate = 1)]
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
gbrt=GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))
importances = gbrt.feature_importances_
features = cancer['feature_names']
indices = np.argsort(importances)
plt.title('Feature Importances')    #[Ranked Feature Importance]
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[9]:


#[Experiment w/ Gradient Boosted Trees (Breast Cancer Dataset - MaxDepth = 1)]
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
gbrt=GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))
importances = gbrt.feature_importances_
features = cancer['feature_names']
indices = np.argsort(importances)
def plot_feature_importances_cancer(model):
       n_features = cancer.data.shape[1] 
       plt.barh(range(n_features), model.feature_importances_, align='center')
       plt.yticks(np.arange(n_features), cancer.feature_names)
       plt.xlabel("Feature importance")
       plt.ylabel("Fetaure")
       plt.ylim(-1, n_features)   
       plot_feature_importances_cancer(gbrt)
       

                     #[Experiment w/ Gradient Boosted Trees (Breast Cancer Dataset - MaxDepth = 0.01)]
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
gbrt=GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))
importances = gbrt.feature_importances_
features = cancer['feature_names']
indices = np.argsort(importances)
def plot_feature_importances_cancer(model):
       n_features = cancer.data.shape[1] 
       plt.barh(range(n_features), model.feature_importances_, align='center')
       plt.yticks(np.arange(n_features), cancer.feature_names)
       plt.xlabel("Feature importance")
       plt.ylabel("Fetaure")
       plt.ylim(-1, n_features)   
       plot_feature_importances_cancer(gbrt)


# In[10]:


#[Kernalized Support Vector Machine (Blobs DataSet)]
X, y = make_blobs(centers = 4, random_state = 8)
y = y%2
mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1"])


# In[11]:


#[Kernalized SVM (Blobs DataSet)]
X, y = make_blobs(centers = 4, random_state = 8)
y = y%2
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend("Class 0", "Class 1")
linear_svm = LinearSVC().fit(X, y)
mglearn.plots.plot_2d_classification(linear_svm, X)
mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


# In[12]:


#[Adding square of second feature]
X_new = np.hstack([X, X[:, 1:]**2])
figure = plt.figure()
ax = Axes3D(figure, elev = -152, azim = -26)     #[visualize in 3D]
mask = y == 0   #[plot first all the points with y = 0, then all with y = 1]
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c = 'b', 
           cmap = mglearn.cm2, s = 60, edgecolor = 'k')
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c = 'r', 
           cmap = mglearn.cm2, s = 60, edgecolor = 'k')
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")


# In[13]:


#[Fitting Linear model to Augmented Data]
linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
figure = plt.figure()
ax = Axes3D(figure, elev = -152, azim = -26)    #[Display in 3D]
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 0].max() + 2, 50)
XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', 
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', 
           cmap=mglearn.cm2, s=60, edgecolor='k')


# In[14]:


ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel, ZZ.ravel()], cmap = mglearn.cm2, alpha = 0.5)
mglearn.disctrete_scatter(X[:,0], X[:,1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


# In[15]:


#[Kernel]
X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X,y)
mglearn.plots.plot_2d_separator(svm, X, eps=0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
sv=svm.support_vectors_   #[plot support vectors]
sv_labels = svm.dual_coef_.ravel() > 0   #[class labels of support vectors are given by the sign of the dual coefficients]
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("Fetaure 0")
plt.ylabel("Fetaure 1")


# In[16]:


#[Tuning SVM Parameters - GAMMA & C]
X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel = 'rbf', C = 10, gamma = 0.1).fit(X,y)
mglearn.plots.plot_2d_separator(svm, X, eps = 0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
sv = svm.support_vectors_    #[plot support vectors]
sv_labels = svm.dual_coef_.ravel() > 0    #[class labels of support vectors are given by the sign of the dual coefficients]
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s = 15, markeredgewidth = 3)
plt.xlabel("Fetaure 0")
plt.ylabel("Fetaure 1")
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C = C, log_gamma=gamma, ax = a)
axes[0, 0].legend(["class 0", "class 1", "sv class 0", "sv class 1"], 
    ncol = 4, loc = (.9, 1.2))


# In[17]:


#[Applying RBF Kernel SVM to Breast Cancer DataSet]
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state = 0)
svc = SVC()
svc.fit(X_train, y_train)
print("Accuracy on the training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on the test set: {:.2f}".format(svc.score(X_test, y_test)))
plt.boxplot(X_train, manage_xticks = False)
plt.yscale("Symlog")
plt.xlabel("Feature Index")
plt.ylabel("Feature Magnitude")


# In[18]:


#[Preprocessing data for SVM - Rescaling Features - Breast Cancer DataSet]
min_on_training = X_train.min(axis = 0)    #[compute the minimum value per feature on the training set]
range_on_training = (X_train - min_on_training).max(axis = 0)    #[compute the range each feature (max-min) on the training set]
X_train_scaled = (X_train - min_on_training)/range_on_training
#[subtract the min, divide by range AND afterward, min = 0 and max = 1 for each feature]
print("Minimum for each feature\n{}".format(X_train_scaled.min(axis = 0)))
print("Maximum for each feature\n{}".format(X_train_scaled.max(axis = 0)))
X_test_scaled = (X_test - min_on_training) / range_on_training
svc = SVC()
svc.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test, y_test)))
svc = SVC(C = 1000)    #[adjusting C for GAMMA]
svc.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))



# In[19]:


#Multilayer Perceptrons
display(mglearn.plots.plot_logistic_regression_graph())    #[Vizualization of logistic regression]
display(mglearn.plots.plot_single_hidden_layer_graph())    #[Vizualization of MLP]


# In[20]:


#[Features of RELU and TANH]
line = np.linspace(-3, 3, 1000)
plt.plot(line, np.tanh(line), label = 'tanh')
plt.plot(line, np.maximum(line, 0), label = 'relu')
plt.legend(loc = "best")
plt.xlabel("x")
plt.ylabel("relu(x), tanh(x)")


# In[21]:


#[Applying MLP - TWO MOONS DATASET]
X, y = make_moons(n_samples = 100, noise = 0.25, random_state = 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42)
mlp = MLPClassifier(solver = 'lbfgs', random_state = 0).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill = True, alpha = .3)
mglearn.discrete_scatter(X_train[:, 0],X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


# In[22]:


#[Reducing hidden layer size]
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)    # split the wave dataset into training and a test set
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10]).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0],X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


# In[23]:


#[Adding 2 hidden layers (RELU)]
X, y = make_moons(n_samples = 100, noise = 0.25, random_state = 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42)    # split the wave dataset into training and a test set
mlp = MLPClassifier(solver = 'lbfgs', random_state = 0, hidden_layer_sizes = [10, 10]).fit(X_train, y_train)    # using two hidden layers, with 10 units each
mglearn.plots.plot_2d_separator(mlp, X_train, fill = True, alpha = .3)
mglearn.discrete_scatter(X_train[:, 0],X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


  #[Adding 2 hidden layers (TANH)]
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)    #[split the wave dataset into training and a test set]

mlp = MLPClassifier(solver='lbfgs', activation ='tanh', random_state=0, hidden_layer_sizes=[10, 10]).fit(X_train, y_train)    #[using two hidden layers, with 10 units each]
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0],X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


# In[24]:


#[Using L2 regularization]
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)    #[split the wave dataset into training and a test set]
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for axx, n_hidden_nodes in zip(axes, [10, 100]):
    for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
        mlp = MLPClassifier(solver='lbfgs', random_state=0, 
                            hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], alpha=alpha)
        mlp.fit(X_train, y_train)
        mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
        mglearn.discrete_scatter(X_train[:, 0],X_train[:, 1], y_train, ax=ax)
        ax.set_title("n_hidden=[{}, {}]\n alpha={:.4f}".format(
                    n_hidden_nodes, n_hidden_nodes, alpha))


# In[25]:


#[Effects of random initialization no weights]
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42)    #[split the wave dataset into training and a test set]
import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(2, 4, figsize = (20, 8))
for i, ax in enumerate(axes.ravel()):
    mlp = MLPClassifier(solver = 'lbfgs', random_state = i, 
                            hidden_layer_sizes = [100, 100])
    mlp.fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
    mglearn.discrete_scatter(X_train[:, 0],X_train[:, 1], y_train, ax=ax)


# In[ ]:


#[Applying MLP (Breast Cancer DataSet)]
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 0)
training_accuracy = []
test_accuracy = []
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))


#Rescaling Breast Cancer DataSet
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
training_accuracy = []
test_accuracy = []
mean_on_train = X_train.mean(axis=0)    #[compute the mean value per feature on the training set]
std_on_train = X_train.std(axis=0)    #[compute the standard deviation of each feature on the training set]
X_train_scaled = (X_train - mean_on_train) / std_on_train    #[subtract the mean, and scale by inversse standard deviation & #afterward, mean=0 and std=1]
X_test_scaled = (X_test - mean_on_train) / std_on_train    #[use THE SAME transformation (using training mean and std) on the test set]
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))



# In[ ]:


#[Setting MAX number of iterations]
# Generate Dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, random_state = 0)
training_accuracy = []
test_accuracy = []
mean_on_train = X_train.mean(axis = 0)    #[compute the mean value per feature on the training set]
std_on_train = X_train.std(axis = 0)    #[compute the standard deviation of each feature on the training set]
X_train_scaled = (X_train - mean_on_train) / std_on_train    #[subtract the mean, and scale by inversse standard deviation & #afterward, mean=0 and std=1]
X_test_scaled = (X_test - mean_on_train) / std_on_train    #[use THE SAME transformation (using training mean and std) on the test set]
mlp = MLPClassifier(max_iter = 1000, random_state = 42)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))


# In[ ]:


#[Looking at importance of weights]
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, random_state = 0)
training_accuracy = []
test_accuracy = []
mean_on_train = X_train.mean(axis = 0)    #[compute the mean value per feature on the training set]
std_on_train = X_train.std(axis = 0)    #[compute the standard deviation of each feature on the training set]
X_train_scaled = (X_train - mean_on_train) / std_on_train    #[subtract the mean, and scale by inversse standard deviation & #afterward, mean=0 and std=1]
X_test_scaled = (X_test - mean_on_train) / std_on_train    #[use THE SAME transformation (using training mean and std) on the test set]
mlp = MLPClassifier(max_iter = 1000, random_state = 42)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
plt.figure(figsize = (20, 5))
plt.imshow(mlp.coefs_[0], interpolation = 'none', cmap = 'viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()


# In[ ]:


#[Running a Neural Network w/ tensorflow]
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
plt.imshow(np.reshape(mnist.train.images[8], [28, 28]), cmap='gray')
plt.show()
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100
#[Network parameters]
n_hidden_1 = 10 #[1st layer number of neuwrons]
n_hidden_2 = 10 #[2nd layer of neurons]
num_input = 784 #[MNIST data input (img shape: 28*28)]
num_classes = 10 #[MNIST total classes (0-9 digits)]
# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])


# In[ ]:


#[Store layers weight & bias]
weights = {
        'h1':
            tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2':
            tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out':
            tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}

biases = {
            'b1':
                tf.Variable(tf.random_normal([n_hidden_1])),
            'b2':
                tf.Variable(tf.random_normal([n_hidden_2])),
            'out':
                tf.Variable(tf.random_normal([num_classes]))
}


# In[ ]:


#[Create model]
def neural_net(x):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])    #[Hidden fully connected layer with 10 neurons]
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])    #[Hidden fully connected layer with 10 neurons]
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']    #[Output fully connected layer with a neuron for each class]
        return out_layer
logits = neural_net(X)    #[Construct model]
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))    #[Define loss and optimizer]
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))    #[Evaluate model (with test logits, for dropout to be disabled)]
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()    #[Initialize the variables (i.e. assign their default value)]  


# In[ ]:


#[Decision Functions]   
X, y = make_circles(noise=0.25, factor=0.5, random_state=1)
y_named = np.array(["blue", "red"])[y]    #[We rename the classes "blue" and "red" for illustration purposes.]
#[split the wave dataset into training and a test set]
#[We can call train_test_split with arbitrarily many arrays]
#[all will be split in a consistent manner]
X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(X, y_named, y, random_state=0)
gbrt = GradientBoostingClassifier(random_state=0)    #[build the gradient boosting model]
gbrt.fit(X_train, y_train_named)
print("X_test.shape: {}".format(X_test.shape))
print("Decision function shape: {}".format(gbrt.decision_function(X_test).shape))
print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6]))    #[Show the first few entires of decision_function]
print("Thresholded decision function:\n{}".format(gbrt.decision_function(X_test) > 0))
print("Predictions: \n{}".format(gbrt.predict(X_test)))
greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)    #[make the boolean True/False into 0 and 1]
pred = gbrt.classes_[greater_zero]    #[use 0 and 1 as indices into classes]
print("pred is equal to predictions: {}".format(np.all(pred == gbrt.predict(X_test))))    #[pred is the same as the output of gbrt.predict]
decision_function = gbrt.decision_function(X_test)
print("Decision function minimum: {:.2f} maximum: {:.2f}".format(
        np.min(decision_function), np.max(decision_function)))



fig, axes = plt.subplots(1, 2, figsize=(13, 5))
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1], alpha=.4, cm=mglearn.ReBl)
for ax in axes:
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=ax)    #[plot training and test points]
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers='o', ax=ax)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
cbar = plt.colorbar(scores_image, ax=axes.tolist())
axes[0].legend(["test class 0", "test class 1", "Train class 0", "Train class 1"], ncol=4, loc=(.1, 1.1))


# In[ ]:


#[Predicting Probabilities]
print("Shape of probabilities: {}".format(gbrt.predict_proba(X_test).shape))
print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test[:6])))    #[show the first few entries of predict_probability]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1], alpha=.4, cm=mglearn.ReBl, function='predict_proba')
for ax in axes:
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=ax)     #[plot training and test points]   
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers='o', ax=ax)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
cbar = plt.colorbar(scores_image, ax=axes.tolist())
axes[0].legend(["test class 0", "test class 1", "Train class 0", "Train class 1"], ncol=4, loc=(.1, 1.1))


# In[ ]:


#[Uncertainty in Multiclass Classification]
iris=datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
#[build the gradient boosting model]
gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(X_train, y_train)
print("Decision function shape: {}".format(gbrt.decision_function(X_test).shape))
#[Show the first few entires of decision_function]
print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6, :]))
print("Argmax of decision function:\n{}".format(np.argmax(gbrt.decision_function(X_test), axis=1)))
print("Predictions: \n{}".format(gbrt.predict(X_test)))
#[Show the first few entires of predict_proba]
print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test)[:6, :]))
#[Show that sums across rows are one]
print("Sums: {}".format(gbrt.predict_proba(X_test)[:6].sum(axis=1)))
print("Argmax of predicted probabilities:\n{}".format(np.argmax(gbrt.predict_proba(X_test), axis=1)))
print("Predictions: \n{}".format(gbrt.predict(X_test)))

