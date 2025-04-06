#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing Necessary Libraires
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib 


# In[6]:


# Loading Preprocessed Data
X_train = pd.read_csv(r"C:\Users\youg_\machine-learning-introduction-YougOsti\data\X_train\X_train.csv")
y_train = pd.read_csv(r"C:\Users\youg_\machine-learning-introduction-YougOsti\data\y_train\y_train.csv")


# In[7]:


# Checkinf for synopsis of X_train data
X_train.head()


# In[8]:


# Checking for synopsis of y_train data
y_train.head()


# In[11]:


# Chosing mode to train data
model = LinearRegression()
# Training data
model.fit(X_train, y_train)


# In[13]:


# Saving the trained data
import os
os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/linear_regression_model.pkl')


# In[ ]:




