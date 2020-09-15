
# coding: utf-8

# In[1]:


import os
import tarfile
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
from http import HTTPStatus


# In[21]:


import matplotlib
import pandas as pd
import requests
import seaborn
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt


# In[3]:


sklearn_housing_bunch = fetch_california_housing("~/data/sklearn_datasets/")


# In[4]:


print(sklearn_housing_bunch.DESCR)


# In[9]:


print(sklearn_housing_bunch.feature_names)


# In[6]:


sklearn_housing = pd.DataFrame(sklearn_housing_bunch.data,
                              columns = sklearn_housing_bunch.feature_names)
print(sklearn_housing)


# In[7]:


def get_data():
    #Gets the data from GitHub and uncompresses it
    if os.path.exists(Data.target):
        return
    
    os.makedirs(Data.source_slug, exist_ok=True)
    response = requests.get(Data.url, stream = True)
    assert response.status_code == HTTPStatus.OK
    with open(Data.source, "wb") as writer:
        for chunk in response.itr_content(chunk_size = Data.chunk_size):
            writer.write(chunk)
        assert os.path.exists(Data.source)
        compressed = tarfile.open(Data.source)
        compressed.extractall(Data.target_slug)
        compressed.close()
        assert os.path.exists(Data.target)
        return


# In[34]:


print(sklearn_housing.AveRooms.head())


# In[17]:


sklearn_housing.hist('AveBedrms',bins=5)
sklearn_housing.hist('MedInc',bins=5)
sklearn_housing.hist('AveRooms',bins=5)
sklearn_housing.hist('HouseAge',bins=5)
sklearn_housing.hist('Population',bins=5)
sklearn_housing.hist('AveOccup',bins=5)
sklearn_housing.hist('Latitude',bins=5)
sklearn_housing.hist('Longitude',bins=5)

