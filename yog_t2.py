#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt






# In[8]:


iris = datasets.load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target


# In[9]:


print("First 10 rows of the Iris dataset:")
print(data.head(10))


# In[10]:


print("Number of instances:", data.shape[0])
print("Number of features:", data.shape[1] - 1)
print("Target classes:", iris.target_names)


# In[16]:


plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Heatmap Correlation")
plt.show()




# In[19]:


X = data.iloc[ :-1]
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, test_size=0.8)



# In[17]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[18]:


model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# In[21]:


y_pred = model.predict()
print("Model Performance on Test Set:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))


# In[ ]:








