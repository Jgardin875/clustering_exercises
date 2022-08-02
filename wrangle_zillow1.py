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


url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/zillow'


# In[3]:


def new_zillow_data():
    return pd.read_sql('''SELECT *
FROM properties_2017 p
LEFT JOIN propertylandusetype t USING (propertylandusetypeid)
LEFT JOIN airconditioningtype a USING (airconditioningtypeid)
LEFT JOIN buildingclasstype b USING (buildingclasstypeid)
LEFT JOIN architecturalstyletype ar USING (architecturalstyletypeid)
RIGHT JOIN predictions_2017 pr USING (parcelid)
WHERE t.propertylandusedesc = 'Single Family Residential'
AND pr.transactiondate LIKE "2017%%"
''', url)


import os

def get_zillow_data():
    filename = "zillow.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col = 0)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df_zillow = new_zillow_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df_zillow.to_csv(filename)

        # Return the dataframe to the calling code
        return df_zillow


# In[4]:


def nulls_by_col(df):
    num_missing = df.isnull().sum()
    prcnt_miss = (num_missing/df.shape[0]) * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing.sort_values(by='num_rows_missing', ascending=False)


# In[5]:


def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = (num_missing/df.shape[1]) * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)


# In[6]:


def handle_missing_values(df, fraction_required_cols, fraction_required_rows):
    #fraction_required_cols * len(df.index)
    #       take the fraction of required columns and mulitply by the number of rows(the number of values that will need to be required by the column is made up of rows)
    #round(fraction_required_cols * len(df.index), 0)
    #       round to zero decimal places
    #int()
    #       convert to integer
    threshold = int(round(fraction_required_cols * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    
    #absence of row values is reflected in the absence of the nubmer of possible boxes, aka number of columns
    threshold = int(round(fraction_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    return df


# In[7]:


def remaining_nulls(df):
    
    df.regionidcity = df.regionidcity.fillna(value = df.regionidcity.mode()[0]) #fill regionidcity with mode
    
    df = df[df.lotsizesquarefeet < 3_000_000.0] # delete repeating values that are unlikely to be credible
    
    df.lotsizesquarefeet = df.lotsizesquarefeet.fillna(value = df.lotsizesquarefeet.median()) #fill nulls with medain for lotsizesquarefeet

    #df.finishedsquarefeet12 = df.finishedsquarefeet12.fillna(value = df.finishedsquarefeet12.median()) # commented out due to high similairty with sqft column but has more nulls than sqft column

    df.yearbuilt = df.yearbuilt.fillna(value = df.yearbuilt.mode()[0])
    
    df.regionidzip = df.regionidzip.fillna(value = df.regionidzip.mode()[0])
    # delete anything with tax values that can be used in calculating target variable logerror
    df = df.drop(columns = ['structuretaxvaluedollarcnt',  'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxamount', 'fullbathcnt', 'calculatedbathnbr'])

    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.fillna(value = df.calculatedfinishedsquarefeet.median())
    # drop remaining 63 nulls in censustractandblock
    df = df.dropna()
    return df


# In[8]:


def split_data(df):
    train_validate, test = train_test_split(df, test_size=.2, random_state=51)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=51)
    return train, validate, test


# In[17]:


def prep_zillow(df, prop_req_cols, prop_req_rows):
    # drop redundant ids, roomcnt appears inaccurate (lots of zeros in rows that have non-zero bed and bath), finishedsqft12 similar to sqft, assessmentyear is the same for all rows
    df = df.drop(columns = ['id', 'propertylandusedesc', 'propertylandusetypeid', 'roomcnt', 'finishedsquarefeet12', 'assessmentyear'])
    df = df.sort_values('transactiondate').drop_duplicates('parcelid',keep='last')
    
    df = df[(df.bathroomcnt <= 6) & (df.bathroomcnt >= 1)]
    df = df[(df.bedroomcnt <= 6) & (df.bedroomcnt >= 1)]
    
    df = handle_missing_values(df, prop_req_cols, prop_req_rows)
    df = remaining_nulls(df)
    df.rename(columns = {'bedroomcnt': 'bed', 'bathroomcnt': 'bath', 'calculatedfinishedsquarefeet' : 'sqft', 'regionidzip':'zip', 'lotsizesquarefeet': 'lot_sqft'}, inplace = True)
    
    train, validate, test = split_data(df)
    
    return df, train, validate, test


# In[18]:


df = new_zillow_data()


# In[19]:


df, train, validate, test = prep_zillow(df, .66, .75)


# In[20]:


df.head()


# In[ ]:




