
# coding: utf-8

# In[1]:


import sklearn.datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import mglearn as mglearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.decomposition import PCA
from numpy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_predict
from sklearn.manifold import TSNE


# In[2]:


#                                                   Question 1
print("Question 1")
print()
print()
print()


# In[3]:


wine = load_wine()
df = pd.DataFrame(wine.data,columns=wine.feature_names)
df['target'] = pd.Series(wine.target)
X_train, X_test, y_train, y_test = train_test_split(wine.data, 
                                                    wine.target, 
                                                    stratify = wine.target, 
                                                    random_state = 42)
df.head()
print()
print(wine.target_names)


# In[4]:


wine.DESCR


# In[5]:


print( X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[6]:


#Scaling Data#
print("Scaling Data")
print('')
scaler = MinMaxScaler()
scaler.fit(X_train)
MinMaxScaler(copy=True, feature_range=(0, 1))
X_train_scaled = scaler.transform(X_train)   # transform data
print("transformed shape: {}".format(X_train_scaled.shape))    # print dataset properties before and after scaling
print('')
print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print('')
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
print('')
print("per-feature minimum after scaling:\n {}".format(X_train_scaled.min(axis=0)))
print('')
print("per-feature maximum after scaling:\n {}".format(X_train_scaled.max(axis=0)))


# In[7]:


X_test_scaled = scaler.transform(X_test)
#print dataset properties before and after scaling
print("transformed shape: {}".format(X_test_scaled.shape))
print('')
print("per-feature minimum before scaling:\n {}".format(X_test.min(axis=0)))
print('')
print("per-feature maximum before scaling:\n {}".format(X_test.max(axis=0)))
print('')
print("per-feature minimum after scaling:\n {}".format(X_test_scaled.min(axis=0)))
print('')
print("per-feature maximum after scaling:\n {}".format(X_test_scaled.max(axis=0)))


# In[8]:


X, _ = make_blobs(n_samples = 50, centers = 5, random_state = 4, cluster_std = 2)
X_train, X_test = train_test_split(X, random_state = 5, test_size = 0.1)


# In[9]:


#plot training and test sets
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=mglearn.cm2(0), label="Training set", s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^', c=mglearn.cm2(1), label='Test set', s=60)
axes[0].legend(loc='upper left')
axes[0].set_title("Original Data")

#scale data using MinMaxScalar
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#visualize the properly scaled data
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label="Training set", s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=mglearn.cm2(1), label="Test set", s=60)
axes[1].set_title("Scaled Data")

for ax in axes:
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    fig.tight_layout()


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, stratify = wine.target, random_state=66)
svm = SVC(C=100)
svm.fit(X_train, y_train)
print("Test set accuracy: {:.2f}".format(svm.score(X_test, y_test)))
print('')
print("Preprocessing using 0-1 scaling with MinMaxScaler")
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('')
print("Learn an SVM on scaled training data:")
svm.fit(X_train_scaled, y_train)
print("Scoring on scaled test set")
print("Scaled test set accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))


# In[11]:


print("Test set accuracy: {:.2f}".format(svm.score(X_test, y_test)))
print('')
print("Preprocessing using 0-1 scaling with MinMaxScaler")
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('')
svm.fit(X_train_scaled, y_train)
print("SVM test set accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))


# In[12]:


#                                                  Question 6
print("Question 2")
print()
print()
print()


# In[13]:


mglearn.plots.plot_pca_illustration()


# In[14]:


fig, axes = plt.subplots (13, 1, figsize = (10, 20))
Class_0 = wine.data[wine.target == 0] 
Class_1 = wine.data [wine.target == 1] 
Class_2 = wine.data [wine.target == 2] 
ax = axes.ravel() 
for i in range (13): 
    _ , bins = np.histogram (wine.data[:,i], bins = 50) 
    ax [i].hist(Class_0 [:,i], bins = bins, color = mglearn.cm3 (0), alpha = .5) 
    ax [i].hist(Class_1 [:,i], bins = bins, color = mglearn.cm3 (2), alpha = .5) 
    ax [i].hist(Class_2 [:,i], bins = bins, color = mglearn.cm3 (1), alpha = .5) 
    ax [i].set_title(wine.feature_names [i]) 
    ax [i].set_yticks(()) 
    ax [0].set_xlabel( "Feature magnitude" ) 
    ax [0].set_ylabel( "Frequency" ) 
    ax [0].legend (["Class_0", "Class_1", "Class_2"], loc = "best")
fig.tight_layout () 


# In[15]:


print("Scale Dataset")
scaler = StandardScaler()
scaler.fit(wine.data)
X_scaled = scaler.transform(wine.data)
print('')

pca = PCA(n_components = 4)
pca.fit(X_scaled)

#transform data onto the first two principal components
X_pca = pca.transform(X_scaled)
print("Original shape: {}".format(str(X_scaled.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))
print('')

#plot first vs. second principal components
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], wine.target)
plt.legend(wine.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")


# In[16]:


scaler = MinMaxScaler(feature_range=[0, 1])
wine_rescaled = scaler.fit_transform(X_scaled)

#Fitting the PCA algorithm with our Data
pca = PCA().fit(wine_rescaled)

#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Wine Dataset Explained Variance')
plt.show()


# In[17]:


#plot second vs. third principal components
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 1], X_pca[:, 2], wine.target)
plt.legend(wine.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("Second principal component")
plt.ylabel("Third principal component")


# In[18]:


#plot secind vs. third principal components, colored by class
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 2], X_pca[:, 3], wine.target)
plt.legend(wine.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("Third principal component")
plt.ylabel("Fourth principal component")


# In[19]:


print("PCA component shape: {}".format(pca.components_.shape))
print("PCA components:\n{}".format(pca.components_))

plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1, 2, 3], ["first component", "Second component", "Third Component", "Third Component"])
plt.colorbar()
plt.xticks(range(len(wine.feature_names)), wine.feature_names, rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")


# In[20]:


plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 3], wine.target)
plt.legend(wine.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Fourth principal component")


# In[21]:


counts = np.bincount(wine.target)
for i, (count, name) in enumerate(zip(counts, wine.target_names)):
    print("{0:8} {1:10}".format(name, count), end="  |  ")
    if (i+1) % 3 == 0:
        print()


# In[22]:


mask = np.zeros(wine.target.shape, dtype=np.bool)
for target in np.unique(wine.target):
    mask[np.where(wine.target == target)[0][:50]] = 1
    
X_wine = wine.data[mask]
y_wine = wine.target[mask]

X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, stratify=y_wine, random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print()
print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test, y_test)))


# In[23]:


mglearn.plots.plot_pca_whitening()

pca = PCA(n_components=4, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)                

print("X_train_pca.shape: {}".format(X_train_pca.shape))
X_train_pca.shape: (1547, 100)


# In[24]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print()
print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test_pca, y_test)))


# In[25]:


#build a PCA model
pca = PCA(n_components=4)
pca.fit(wine.data)

#transform the digits data onto the first two principal components
wine_pca = pca.transform(wine.data)
plt.figure(figsize=(10, 10))
plt.xlim(wine_pca[:, 0].min(), wine_pca[:, 0].max())
plt.ylim(wine_pca[:, 1].min(), wine_pca[:, 1].max())
colors = ["#476A24", "#7851B8", "#BD3430"]
for i in range(len(wine.data)):
    plt.text(wine_pca[i, 0], wine_pca[i, 1], str(wine.target[i]), color = colors[wine.target[i]], fontdict={'weight': 'bold', 'size': 10})
plt.xlabel("First principal component")
plt.ylabel("Second principal component")


# In[26]:


tsne = TSNE(random_state=42)

#use fit_transform instead of fit, as TSNE has no transform method
wine_tsne = tsne.fit_transform(wine.data)
plt.figure(figsize=(10, 10))
plt.xlim(wine_tsne[:, 0].min(), wine_tsne[:, 0].max()+1)
plt.ylim(wine_tsne[:, 1].min(), wine_tsne[:, 1].max()+1)
for i in range(len(wine.data)):

    plt.text(wine_tsne[i, 0], wine_tsne[i, 1], str(wine.target[i]), color = colors[wine.target[i]], fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("t-SNE First principal component")
plt.ylabel("t-SNE Second principal component")


# In[27]:


#                                               Question 6
print("Question 6")
print()
print()
print()


# In[28]:


data = make_blobs(n_samples = 300, n_features = 2, centers = 3, cluster_std = 1.5, random_state=20)
points = data[0]
print(data)


# In[29]:


#Plotting data
print("Plotting data")
plt.scatter(data[0][:,0], data[0][:,1], c = data[1], cmap = 'viridis')
plt.xlim(-15, 15)
plt.ylim(0, 15)


# In[30]:


#KMeans Clustering
print("KMeans Clustering")
kmeans = KMeans(n_clusters = 3)
kmeans.fit(points) #fit
print(kmeans.cluster_centers_) #cluster locations
y_km = kmeans.fit_predict(points) #saving clusters for charts


# In[31]:


print("Plotting KMeans")
plt.scatter(points[y_km == 0,0], points[y_km == 0,1], s=50, c='red')
plt.scatter(points[y_km == 1,0], points[y_km == 1,1], s=50, c='black')
plt.scatter(points[y_km == 2,0], points[y_km == 2,1], s=50, c='blue')


# In[32]:


#Hierarchical Clusering
print("Hierarchical Clusering")
dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))
hc = AgglomerativeClustering(n_clusters=3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(points)


# In[33]:


print("Plotting Hierarchical")
plt.scatter(points[y_hc ==0,0], points[y_hc == 0,1], s=50, c='red')
plt.scatter(points[y_hc==1,0], points[y_hc == 1,1], s=50, c='black')
plt.scatter(points[y_hc ==2,0], points[y_hc == 2,1], s=50, c='blue')


# In[34]:


#                                                  Question 7
print("Question 7")
print()
print()
print()


# In[35]:


iris = load_iris()
iris.data.shape


# In[36]:


X = iris.data[:, :2]
y = iris.target
print(X.shape)
print(y.shape)


# In[37]:


plt.scatter(X[:,0], X[:,1], c = y, cmap = 'viridis')
plt.xlabel('Sepal Length', fontsize=10)
plt.ylabel('Sepal Width', fontsize=10)


# In[48]:


kmeans = KMeans(n_clusters = 3, n_jobs = 4, random_state=21)
kmeans.fit(X) #fitting data
centers = kmeans.cluster_centers_
y_km = kmeans.fit_predict(points) #saving clusters for charts
print(centers)


# In[49]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_)
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2], 
                         markers='o', markeredgewidth=6)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")


# In[41]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print("Cluster:\n{}".format(kmeans.labels_))
print("k-Means Prediction:\n{}".format(kmeans.predict(X)))

lr = linear_model.LinearRegression()
y = iris.target

predicted = cross_val_predict(lr, iris.data, y, cv=5)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], ' ', lw=2)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted')
plt.show()

