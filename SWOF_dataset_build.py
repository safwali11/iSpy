#!/usr/bin/env python
# coding: utf-8

# In[1]:


import SWOF_functions_complete as SWOF
import numpy as np


# In[2]:



#### 1. IMPORT VIDEO FOOTAGE FOR FRAMES TRAINING


# In[3]:


video, length, width, height, fsp = SWOF.load_video('crash5.mp4')


# In[4]:


frames = SWOF.split_video_frames(video, length, height, width)


# In[5]:



#### 2. GENERATE TRAINING DATASETS IN CLASS LABELS DIRECTORIES


# In[6]:


#directory must be labeled as "../dataset/training/crash"  
CRASHDIR = '/home/jetson/ENGIN492/data_set/training/crash/'

SWOF_crash_images = SWOF.generate_SWOF_data(CRASHDIR, 'crash5',
                                            frames, N=5,
                                            startframe=68, endframe=105)


# In[7]:


#directory must be labeled as "../dataset/training/nocrash"  
NOCRASHDIR = '/home/jetson/ENGIN492/data_set/training/nocrash/'

SWOF_nocrash_images = SWOF.generate_SWOF_data(NOCRASHDIR, 'crash5',
                                              frames, N=5,
                                              startframe=28, endframe=65)


# In[8]:



#### 3. GENERATE TESTING DATASETS IN RESPECTIVE CLASS LABELS DIRECTORIES


# In[10]:


#directory must be labeled as "../dataset/testing/crash/"  
CRASHTESTDIR = '/home/jetson/ENGIN492/data_set/testing/crash/'
SWOF.generate_testdata(TRAINDIR=CRASHDIR,
                       TESTDIR=CRASHTESTDIR,
                       img_perc=0.1)
 


# In[11]:


#directory must be labeled as "../dataset/testing/nocrash/"  
NOCRASHTESTDIR = '/home/jetson/ENGIN492/data_set/testing/nocrash/'
SWOF.generate_testdata(TRAINDIR=NOCRASHDIR,
                       TESTDIR=NOCRASHTESTDIR,
                       img_perc=0.1)

