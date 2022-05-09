#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libs
import pandas as pd


# # Dataset
# [SP500 Since 1950](https://www.kaggle.com/datasets/benjibb/sp500-since-1950?resource=download)

# In[2]:


df = pd.read_csv('GSPC.csv')


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.drop('Date', axis=1, inplace=True)


# In[10]:


df.sample(5)


# In[11]:


df[-5::]


# In[ ]:


# separar a úlitma linha do dataframe
futuro = df[-1::]


# In[14]:


futuro


# In[15]:


presente = df.drop(df[-1::].index, axis=0)


# In[16]:


presente.tail()


# In[17]:


presente['target'] = presente['Close'][1:len(presente)].reset_index(drop=True)


# In[20]:


presente.tail()


# In[19]:


presente.head()


# In[22]:


prev = presente[-1::].drop('target', axis =1)


# In[23]:


prev


# In[24]:


treino = presente.drop(presente[-1::].index, axis=0)


# In[26]:


treino


# In[31]:


treino.loc[treino['target'] > treino['Close'], 'target'] = 1
treino.loc[treino['target'] != 1, 'target'] = 0
treino['target'] = treino['target'].astype(int)


# In[32]:


treino


# In[37]:


y = treino['target']
X = treino.drop('target', axis=1)


# In[34]:


y


# In[38]:


X


# In[39]:


from sklearn.model_selection import train_test_split
X_tr, X_ts, y_tr, y_ts = train_test_split(X,y, test_size = 0.3)

from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier()
modelo.fit(X_tr, y_tr)

resultado = modelo.score(X_ts, y_ts)
print (f'Acurácia: {resultado}')


# In[40]:


prev


# In[43]:


modelo.predict(prev)


# In[42]:


futuro


# In[44]:


prev['target'] = modelo.predict(prev)


# In[45]:


prev


# In[46]:


futuro


# In[47]:


presente


# In[48]:


presente = presente.append(futuro, sort=True)


# In[49]:


presente.tail()


# In[ ]:




