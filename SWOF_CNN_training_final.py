#!/usr/bin/env python
# coding: utf-8

# In[1]:


import SWOF_functions_complete as SWOF
import keras
import matplotlib.pyplot as plt
import numpy as np


# In[2]:



### 1. LOAD DATASET


# In[3]:


DATAFOLDER = '/home/jetson/complete/dataset/'


# In[4]:


#training dataset
#with 5% validation split

train_generator, X_train = SWOF.CNN_train_gen(DATADIR=DATAFOLDER,
                                              traindatadirname='training',
                                              val_set=0.05)


# In[5]:


#testing dataset

test_generator, X_test = SWOF.CNN_test_gen(DATADIR=DATAFOLDER,
                                           testdatadirname='testing')


# In[6]:


first_image = X_train[0][0][0]
first_label = X_train[0][1][0]


# In[7]:



### 2. SETUP THE NEURAL NETWORK


# In[8]:


model = SWOF.CNN_training(first_image=first_image,
                          first_label=first_label,
                          NUMBER_OF_CLASSES=2)


# In[9]:



### 3. TRAIN THE NEURAL NETWORK


# In[10]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[11]:


history = model.fit(X_train, epochs=50)


# In[12]:



### 4. ANALYZE THE TRAINING MODEL


# In[13]:


SWOF.CNN_analysis_plots(modelfit=history)


# In[14]:



### 5. STORE THE TRAINING MODEL


# In[16]:


# save model and architecture to single file
model.save('CNN_model.h5')
print("Saved model to disk")


# In[ ]:




