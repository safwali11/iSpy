#!/usr/bin/env python
# coding: utf-8

# In[2]:


import SWOF_functions_complete as SWOF
import numpy as np
#%pylab inline


# In[ ]:



#### 1. IMPORT VIDEO FOOTAGE FOR FRAMES TRAINING


# In[3]:


video, length, width, height, fsp = SWOF.load_video('crash5.mp4')


# In[4]:


frames = SWOF.split_video_frames(video, length, height, width)


# In[ ]:



#### 2. GENERATE TRAINING DATASETS IN CLASS LABELS DIRECTORIES


# In[5]:


#directory must be labeled as "../dataset/training/crash"  
CRASHDIR = '/home/jetson/ENGIN492/data_set/training/crash/'

SWOF_crash_images = SWOF.generate_SWOF_data(DATADIR=CRASHDIR,
                                            VIDEONAME='crash5',
                                            frames=frames,
                                            N=5,
                                            startframe=68,
                                            endframe=105)


# In[6]:


#directory must be labeled as "../dataset/training/nocrash"  
NOCRASHDIR = '/home/jetson/ENGIN492/data_set/training/nocrash/'

SWOF_nocrash_images = SWOF.generate_SWOF_data(DATADIR=NOCRASHDIR,
                                              VIDEONAME='crash5',
                                              frames=frames,
                                              N=5,
                                              startframe=28,
                                              endframe=65)


# In[ ]:



#### 3. GENERATE TESTING DATASETS IN RESPECTIVE CLASS LABELS DIRECTORIES


# In[6]:


#directory must be labeled as "../dataset/testing/crash/"  
# CRASHTESTDIR = '/home/jetson/ENGIN492/data_set/testing/crash/'

# crash_testdata_images = SWOF.generate_testdata(DATADIR=CRASHTESTDIR,
#  


# In[7]:


#directory must be labeled as "../dataset/testing/nocrash/"  
# NOCRASHTESTDIR = '/home/jetson/ENGIN492/data_set/testing/crash/'

# nocrash_testdata_images = SWOF.generate_testdata(DATADIR=NOCRASHTESTDIR,
# 


# In[ ]:




