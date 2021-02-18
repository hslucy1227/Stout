#!/usr/bin/env python
# coding: utf-8

# ## Name: Shen Huang
# ## Email: huangs9@uci.edu
# 
# ## Case Study #1

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from sklearn.metrics import roc_curve, auc

plt.style.use('ggplot')


# In[2]:


df = pd.read_csv("D:/OA/Stout/PS_20174392719_1491204439457_log.csv") 


# ## Exploratory Data Analysis

# In[3]:


df.info()


# The dataset has 11 columns and 6362620 rows. The following are the first 10 rows of the dataset.

# In[4]:


df.head(10)


# In[5]:


pd.isnull(df).sum().sum()


# The dataset has no missing values.

# In[6]:


df.describe()


# In[7]:


print('Number of true transaction:')
print(df['isFraud'].value_counts())
print('\n Flagged transaction:')
print(df['isFlaggedFraud'].value_counts())


# As we can see, the number of true fraudulent transaction is 8213 while the number of flagged fraudulent transaction is only 16, where the flags are seriously inconsistent with the facts.

# In[8]:


print('Statistics of transaction type:')
print(df['type'].value_counts())
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
df['type'].value_counts().plot(kind='bar', title='Transaction Type', ax=ax, figsize=(8, 4))
plt.show()


# In[9]:


plt.figure(figsize=(14,4))

plt.subplot(1,2,1)

ax = df.groupby(['type', 'isFraud']).size().plot(kind='bar')   
ax.set_title('Number of true/fraudulent transaction type')
ax.set_xlabel('(Type, IsFraud)')
ax.set_ylabel('Number of transaction')
ax.set_ylim([0, 2500000])

for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))  
    
plt.subplot(1,2,2)    
    
ax = df.groupby(['type', 'isFlaggedFraud']).size().plot(kind='bar') 
ax.set_title('Number of flagged transaction type')
ax.set_xlabel('(Type, IsFlaggedFraud)')
ax.set_ylabel('# of transaction')
ax.set_ylim([0, 2500000])

for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))


plt.show()


# From the above bar charts, we can observe that the transaction of cash_in, debit and payment are all true, over 0.18% transaction of cash out and 0.76% transaction of transfer are fraudulent. The wrong flag mainly occurs in the judgment of the transfer transaction. Therefore, I will focus on the data of transfer transaction in the next step.

# In[10]:


transfer_data = df[df['type'] == 'TRANSFER']


# In[11]:


fig, axs = plt.subplots(1,3, figsize=(15, 4)) #
a = sns.boxplot(x='isFlaggedFraud', y='amount', data=transfer_data, ax=axs[0]) 
axs[0].set_yscale('log')   
 
b = sns.boxplot(x='isFlaggedFraud', y='oldbalanceDest', data=transfer_data, ax=axs[1])  
axs[1].set(ylim=(0, 0.5e8))    
 
c = sns.regplot(x='oldbalanceOrg', y='amount', data=transfer_data[transfer_data['isFlaggedFraud'] ==1], ax=axs[2])
plt.show()


# Observing the relationship between the transfer amount and whether the it is flagged as fraud, we can see that transfer transactions marked as fraud tend to have higher amounts and those accounts have lower initial balance. The interesting note is that there may be a linear relationship between initial balance and transfer amount, the more the original balance of the account, the more it will be transferred out.

# ## Predictive modeling

# Since the transactions of cash_in, debit and payment are all true, I will build two models, Logistic Regression and Decision Tree, to focus on the prediction of transactions of cash out and transfer.

# To encode the transactions type, I set a new variable called typeCategory, typeCategory = 1 indicates transactio is transfer and typeCategory = 0 indicates transactio is cash out. 
# 

# To evaluate the Since the dataset is imbalanced, I make use of AUC to evaluate the performance of models. From the above results, we can see that both Logistic Regression and Decision Tree perform well in the prediction of fraud transactions, they are around 95%., I will split the dataset into 2 parts, where 70% is training data, 30% is test data.

# In[12]:


df2 = df[(df['type'] == 'TRANSFER') | (df['type'] == 'CASH_OUT')]

type_label_encoder = preprocessing.LabelEncoder()   
type_category = type_label_encoder.fit_transform(df2['type'].values)
df2['typeCategory'] = type_category
df2.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud','type'], axis=1, inplace=True)

df2.head(10)


# In[13]:


X = df2[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'typeCategory']]
y = df2['isFraud']


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# ### Logistic Regression

# In[15]:


clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)


# In[16]:


print(metrics.classification_report(y_test, y_pred))


# In[17]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_score[:, 1])  
roc_auc = auc(fpr,tpr)
plt.title('Receiver Operating Characteristic (Decision Tree)')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ### Decision Tree

# In[18]:


clf2 = DecisionTreeClassifier()
clf2 = clf2.fit(X_train,y_train)
y_pred2 = clf2.predict(X_test)
y_pred_score2 = clf2.predict_proba(X_test)


# In[19]:


print(metrics.classification_report(y_test, y_pred2))


# In[20]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_score2[:, 1])  
roc_auc = auc(fpr,tpr)
plt.title('Receiver Operating Characteristic (Decision Tree)')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Since the dataset is imbalanced, I make use of AUC to evaluate the performance of models. From the above results, we can see that both Logistic Regression and Decision Tree perform well in the prediction of fraud transactions, they are around 95%. It is worth noting that Decision Tree has higher precision, recall and f1-score, which indicates that it is more effective in predicting non-fraud transactions.

# The data is imbalanced, models will have a better performance if we can get more sample data.

# In[ ]:




