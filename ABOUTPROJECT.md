# iSpy
Real Time Collision Avoidance Deep Learning Image Processing Alert System

## Table of Contents


## Introduction
This project presents a real-time collision avoidance alert system using deep learning image processing to trigger a microcontroller. The motivation for this project is to integrate the system with first-person-view drone flight controllers to assist drone pilots in avoiding potential collisions. To predict potential collisions in real time, the system inputs real-time velocity vector field calculations into a convolutional neural network, developed and trained on the NVIDIA™ Jetson Nano, a small artificial intelligence computer. A convolutional neural network is a class of deep learning used for image recognition and classification by applying convolution filter layers to learn features in a dataset for future predictions. Vector field calculations are performed by way of optical flow which is the process of detecting and calculating the movement of pixels between consecutive frames. A threshold was set to trigger an alert to the drone flight controller through a universal asynchronous receiver/transmitter, a device for data transmission. The training model achieved 85.6\% accuracy with 33.6\% loss. When the system was tested, it was able to predict and alert with approximately 72\%.
There is still continuous progress on this project to improve the training model for higher accuracy and lower loss, as well as research on the different parameters of the system.

## Set Up


## Design


## Files

