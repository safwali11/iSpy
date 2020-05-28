#!/usr/bin/env python
# coding: utf-8

# In[1]:


import SWOF_functions_complete as SWOF
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import cv2


# In[2]:



### 1. LOAD THE TRAINING MODEL


# In[3]:


# load model
model = load_model('CNN_model.h5')
# summarize model.
model.summary()


# In[4]:



### 2. REAL TIME SWOF


# In[5]:


video = cv2.VideoCapture(0)


# In[ ]:


SWOF_images, frames, predictions = SWOF.rtcalc_SWOF_interm(model=model, 
                                                            video=video, 
                                                            N=5, 
                                                            threshold_val=0.2, 
                                                            iterations_param=25)

