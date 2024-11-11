#!/usr/bin/env python
# coding: utf-8

# In[24]:


# import libraries

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# In[3]:


# load dataset

from sklearn.datasets import load_wine

data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['wine'] = data.target
df.head()


# In[4]:


# describe data

df.info()


# In[5]:


df[['alcohol']].describe()


# In[6]:


df[['proline']].describe()


# In[17]:


df[['malic_acid']].describe()


# In[16]:


df[['ash']].describe()


# In[15]:


df[['alcalinity_of_ash']].describe()


# In[14]:


df[['magnesium']].describe()


# In[13]:


df[['total_phenols']].describe()


# In[12]:


df[['flavanoids']].describe()


# In[11]:


df[['nonflavanoid_phenols']].describe()


# In[10]:


df[['proanthocyanins']].describe()


# In[9]:


df[['color_intensity']].describe()


# In[8]:


df[['hue']].describe()


# In[7]:


df[['od280/od315_of_diluted_wines']].describe()


# In[18]:


# explore potential correlation

df.plot(kind = 'scatter', x = 'proline', y = 'wine')


# In[19]:


df.plot(kind = 'scatter', x = 'malic_acid', y = 'wine')


# In[20]:


df.plot(kind = 'scatter', x = 'alcalinity_of_ash', y = 'wine')


# In[21]:


df.plot(kind = 'scatter', x = 'magnesium', y = 'wine')


# In[22]:


df.plot(kind = 'scatter', x = 'flavanoids', y = 'wine')


# In[23]:


df.plot(kind = 'scatter', x = 'color_intensity', y = 'wine')


# In[32]:


# split data into training and test sets

response = 'wine'
y = df[[response]]
y


# In[34]:


x = data.data
x


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)


# In[36]:


# Standardize data

scaler = StandardScaler()

# Fit on training set only.
scaler.fit(X_train)

# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[37]:


# multi_class is specifying one versus rest
clf = LogisticRegression(solver='liblinear',
                         multi_class='ovr', 
                         random_state = 0)

clf.fit(X_train, y_train)
print('Training accuracy:', clf.score(X_train, y_train))
print('Test accuracy:', clf.score(X_test, y_test))


# In[ ]:




