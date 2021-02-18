#!/usr/bin/env python
# coding: utf-8

# ## Name: Shen Huang
# ## Email: huangs9@uci.edu
# 
# ## Case Study #2

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("D:/OA/Stout/casestudy.csv") 


# In[3]:


df.head(10)


# ### Total revenue for the current year

# In[4]:


print(df.groupby(['year'])['net_revenue'].sum())


# ### New Customer

# In[5]:


customer2015 = df[df['year'] == 2015]['customer_email'].values.tolist()
customer2016 = df[df['year'] == 2016]['customer_email'].values.tolist()
customer2017 = df[df['year'] == 2017]['customer_email'].values.tolist()


# In[6]:


newcustomer2016 = []
newcustomer2016 = [item for item in customer2016 if item not in customer2015]
newcustomer2017 = []
newcustomer2017 = [item for item in customer2017 if item not in customer2016]


# In[ ]:


df.loc[df['customer_email'].isin([newcustomer2016])].groupby(['year'])['net_revenue'].sum())


# ### Lost Customers

# In[ ]:


lostcustomer2015 = []
lostcustomer2015 = [item for item in customer2015 if item not in customer2016]
lostcustomer2016 = []
lostcustomer2016 = [item for item in customer2016 if item not in customer2017]


# ### Existing Customer Revenue Current Year

# In[ ]:


existingcustomer2016 = [item for item in customer2016 if item not in newcustomer2016]
existingcustomer2017 = [item for item in customer2017 if item not in newcustomer2017]


# In[ ]:





# In[ ]:




