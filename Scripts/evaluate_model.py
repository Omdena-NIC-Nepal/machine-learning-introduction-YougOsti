#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


# Loading Trained model
model = joblib.load('../models/linear_regression_model.pkl')

# Loading test data
X_test = pd.read_csv(r"C:\Users\youg_\machine-learning-introduction-YougOsti\Notebooks\data\X_test.csv")
y_test = pd.read_csv(r"C:\Users\youg_\machine-learning-introduction-YougOsti\Notebooks\data\y_test.csv")


# In[3]:


# Making prediciton
y_pred = model.predict(X_test)


# In[4]:


# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[5]:


print(f'Mean Square Error is:  {mse}')


# In[6]:


print(f'R-Squared is:  {r2}')


# #### Ploting the residuals

# In[7]:


residuals = y_test.values.flatten() - y_pred.flatten()

plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True, color = 'blue')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.show()


# In[8]:


# Loading model and test data
model = joblib.load('models/linear_regression_model.pkl')
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

