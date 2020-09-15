#!/usr/bin/env python
# coding: utf-8

# In[4]:


#LIBRARIES
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


x=np.array([[1,2,3], [1,2,3]])
print("x:\n{}".format(x))


# In[8]:


eye = np.eye(4)
print("Numpy Array:\n", eye)


# In[13]:


sparse_matrix = sparse.csr_matrix(eye)
print("\nScipy sparse CSRin matrix: \n", sparse_matrix)


# In[12]:


data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices,col_indices)))
print("COO representation: \n", eye_coo)


# In[17]:


x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker = "x")


# In[18]:


data = {'Name': ["John", "Anna", "Peter", "Linda"], 'Locations': ["New York", "Paris", "Berlin", "London"], 
        'Age': [24, 13, 53, 33]}
data_pandas = pd.DataFrame(data)
display(data_pandas)


# In[19]:


display(data_pandas[data_pandas. Age>30])


# In[ ]:




