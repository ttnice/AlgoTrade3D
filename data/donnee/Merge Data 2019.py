#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np


# In[57]:


df = pd.read_csv('EURUSD-2019.csv', names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Bob'])
df.drop('Bob', axis=1, inplace=True)


# In[12]:


# df.to_csv('EURUSD-2019.csv', float_format='%g', index=False)


# In[114]:


names = ['EURJPY', 'EURCHF', 'CHFJPY', 'USDCHF', 'USDCAD']
actions = ['Open', 'High', 'Low', 'Close']
for name in names:
    print(f'Import {name}')
    _df = pd.read_csv(f'{name}-2019.csv', names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Bob'])
    for idx, row in df.iterrows():
        __df = _df.loc[_df.Date == row.Date].loc[_df.Time == row.Time]
        try:
            for action in actions:
                df.loc[idx, f'{name}_{action}'] = list(__df[action])[0]
        except:
            print(f'Error {idx} for {name} : {row.Date} {row.Time}')
df.to_csv('Merged_2019.csv', float_format='%g', index=False)


# In[113]:





# In[110]:




