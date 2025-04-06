#!/usr/bin/env python
# coding: utf-8

# ## Data Preprocessing (Boston Housing Data)
# 

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[2]:


# Load the data
data = pd.read_csv(r"C:\Users\youg_\machine-learning-introduction-YougOsti\BostonHousing.csv")
data.head(2)


# In[3]:


data.info()


# In[4]:


data.describe()


# ### Handeling Outliers

# In[5]:


# Detecting outliers using IQR
def detect_outliers_iqr(data):
    outliers = {}
    for col in data.select_dtypes(include=['number']).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col].count()
        if outlier_count > 0:
            outliers[col] = outlier_count
    return outliers

# Detect outliers
outliers_found = detect_outliers_iqr(data)
print("Outlier counts per column:", outliers_found)


# In[6]:


# Removing Outliers using isolation forest


# In[7]:


from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)  # Assuming 5% of data are outliers
outliers = iso.fit_predict(data.select_dtypes(include=['number']))

data_cleaned = data[outliers == 1] 

print("Rows before removing outliers:", data.shape[0])
print("Rows after removing outliers:", data_cleaned.shape[0])


# ### Standardize numerical features

# In[8]:


scaler = StandardScaler()
num_cols = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio', 'b', 'lstat']
data[num_cols] = scaler.fit_transform(data[num_cols])

# Splitting the dataset
X = data.drop(columns=['medv'])
y = data['medv']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


# Saving the preprocessed data
import os
os.makedirs("./data", exist_ok=True)

# Saving the datasets 
X_train.to_csv("./data/X_train.csv", index=False)
X_test.to_csv("./data/X_test.csv", index=False)
y_train.to_csv("./data/y_train.csv", index=False)
y_test.to_csv("./data/y_test.csv", index=False)

