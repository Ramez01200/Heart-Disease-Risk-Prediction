#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Visual Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Importing necessary libraries for encoding
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Importing necessary library for scaling
from sklearn.preprocessing import StandardScaler

# Importing necessary library for train-test split
from sklearn.model_selection import train_test_split

# Importing necessary libraries for model development and evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Importing necessary library for hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV


# In[31]:


data = pd.read_csv("C:/Users/Ramez/Downloads/archive (1)/CVD_cleaned.csv")


# In[32]:


data.head()


# In[33]:


data.info()


# In[34]:


data.isnull().sum()


# In[35]:


data.describe()


# In[36]:


cols = ['General_Health','Exercise','Heart_Disease','Skin_Cancer','Other_Cancer','Depression','Diabetes','Arthritis','Sex','Age_Category','Smoking_History']


# In[37]:


for i in cols:
    print(data[i].value_counts())
    print()


# In[38]:


cols_tmp = ['Exercise','Heart_Disease','Skin_Cancer','Other_Cancer','Depression','Arthritis','Smoking_History']


# In[39]:


data[cols_tmp] = data[cols_tmp].replace({
    'Yes':1,
    'No':0
})

data[cols_tmp]


# In[40]:


data['General_Health'].value_counts()


# In[41]:


data['General_Health'] = data['General_Health'].replace({
    'Very Good':5,
    'Good':4,
    'Excellent':3,
    'Fair':2,
    'Poor':1
})


# In[42]:


data['General_Health'].value_counts()


# In[43]:


data['Sex'] = data['Sex'].replace({
    'Male':0,
    'Female':1
})


# In[44]:


data['Sex'].value_counts()


# In[45]:


data['Checkup'].value_counts()


# In[46]:


data['Checkup'] = data['Checkup'].replace({
    'Never':0,
    'Within the past year':1,
    'Within the past 2 years':2,
    'Within the past 5 years':3,
    '5 or more years ago':4
})

data['Checkup'].value_counts()


# In[47]:


data['Age_Category'].value_counts()


# In[48]:


age_category_mapping = {
    '18-24': 0,
    '25-29': 1,
    '30-34': 2,
    '35-39': 3,
    '40-44': 4,
    '45-49': 5,
    '50-54': 6,
    '55-59': 7,
    '60-64': 8,
    '65-69': 9,
    '70-74': 10,
    '75-79': 11,
    '80+': 12
}
data['Age_Category'] = data['Age_Category'].map(age_category_mapping)    


# In[49]:


data.head()


# In[50]:


data['Age_Category'].value_counts()


# In[51]:


data['Age_Category'].info()


# In[52]:


data.head(30)


# In[53]:


# Columns for scaling
scale_cols = ["Height_(cm)", "Weight_(kg)", "BMI", "Alcohol_Consumption", 
              "Fruit_Consumption", "Green_Vegetables_Consumption", "FriedPotato_Consumption"]

# Perform scaling
scaler = StandardScaler()
data[scale_cols] = scaler.fit_transform(data[scale_cols])

data.head()


# In[54]:


# Mapping for Diabetes
diabetes_mapping = {
    'No': 0, 
    'No, pre-diabetes or borderline diabetes': 0, 
    'Yes, but female told only during pregnancy': 1,
    'Yes': 1
}
data['Diabetes'] = data['Diabetes'].map(diabetes_mapping)


# In[55]:


data.head()


# In[56]:


# Defining the features (X) and the target (y)
X = data.drop("Heart_Disease", axis=1)
y = data["Heart_Disease"]

# Performing the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[57]:


# Defining the function to apply models
def apply_model(model, X_train, y_train, X_test, y_test):
    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
        # Compute ROC curve and ROC area
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return accuracy, precision, recall, f1, fpr, tpr, roc_auc

# Defining the models
models = [
    ("Logistic Regression", LogisticRegression(random_state=42, max_iter=500)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("Random Forest", RandomForestClassifier(random_state=42))
    
]
# Applying the models and storing the results
results = []
roc_curves = []

for name, model in models:
    accuracy, precision, recall, f1, fpr, tpr, roc_auc = apply_model(model, X_train, y_train, X_test, y_test)
    results.append((name, accuracy, precision, recall, f1))
    roc_curves.append((name, fpr, tpr, roc_auc))

results, roc_curves

