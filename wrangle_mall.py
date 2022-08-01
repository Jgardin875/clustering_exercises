#!/usr/bin/env python
# coding: utf-8

# In[1]:


# personally made imports
import env

# typical imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings("ignore")


# In[2]:


url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/mall_customers'


# In[3]:


def new_mall_data():
    return pd.read_sql('''SELECT *
FROM customers
''', url)


import os

def get_mall_data():
    filename = "mall.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col = 0)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df_mall = new_mall_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df_mall.to_csv(filename)

        # Return the dataframe to the calling code
        return df_mall


# In[4]:


def split_mall_data(df):
    train_validate, test = train_test_split(df, test_size=.2, random_state=51)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=51)
    return train, validate, test


# In[ ]:





# In[5]:


def prep_mall_data(df):
    
    #create dummy columns for catagorical varaibles
    dummy_df = pd.get_dummies(df['gender'], dummy_na=False, drop_first= True)
    df = pd.concat([df, dummy_df], axis=1)

    df = df.drop(columns = 'gender')
    
    train, validate, test = split_mall_data(df)
    
    return df, train, validate, test


# In[6]:


from sklearn.preprocessing import MinMaxScaler

#Define function to scale all data based on the train subset
def mms_scale_data(train, validate, test):
    
    mms_cols = ['age', 'annual_income']
    
    train_mms = train.copy()
    validate_mms = validate.copy()
    test_mms = test.copy()
    
    mms = MinMaxScaler()
    
    mms.fit(train[mms_cols])
    
    train_mms[mms_cols] = mms.transform(train[mms_cols])
    validate_mms[mms_cols] = mms.transform(validate[mms_cols])
    test_mms[mms_cols] = mms.transform(test[mms_cols])
    
    return train_mms, validate_mms, test_mms


# In[ ]:





# In[7]:


#skip for now, work on later


# In[8]:


def data_leakage(train, validate, test, target_variable):    
    x_train = train.drop(columns=[target_variable])
    y_train = train.target_variable

    x_validate = validate.drop(columns=[target_variable])
    y_validate = validate.target_variable

    x_test = test.drop(columns=[target_variable])
    y_test = test.target_variable
    
    print(f'x_train {x_train}, y_train {y_train}, x_validate {x_validate}, y_validate {y_validate} x_test {x_test}, y_test {y_test}')
                 
                 
                 
                 
                 
                 


# In[9]:


df = get_mall_data()


# In[10]:


df, train, validate, test = prep_mall_data(df)


# In[11]:


df.head()


# In[12]:


#xt, yt, xv, yv, xtest, ytest = data_leakage(train, validate, test, df.spending_score)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




