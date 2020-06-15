# iSpy
## Real Time Collision Avoidance Deep Learning Image Processing Alert System

Project Proceedings PDF May 2020

**Version 1.0.0**

## Table of Contents
1. Introduction
2. Hardware and Software Setup
3. Design
4. Files Description
5. Contributors

---

## 1. Introduction
This project presents a real-time collision avoidance alert system using deep
learning image processing to trigger a microcontroller. The motivation for this
project is to integrate the system with first-person-view drone flight
controllers to assist drone pilots in avoiding potential collisions.
To predict potential collisions in real time, the system inputs real-time
velocity vector field calculations into a convolutional neural network,
developed and trained on the NVIDIA™ Jetson Nano, a small artificial
intelligence computer. A convolutional neural network is a class of deep
learning used for image recognition and classification by applying convolution
filter layers to learn features in a dataset for future predictions. Vector
field calculations are performed by way of optical flow which is the process of
detecting and calculating the movement of pixels between consecutive frames.
A threshold was set to trigger an alert to the drone flight controller through
a universal asynchronous receiver/transmitter, a device for data transmission.
The training model achieved 85.6\% accuracy with 33.6\% loss. When the system
was tested, it was able to predict and alert with approximately 72\%.

There is project is in prgress to improve they system by improving the training
model for higher accuracy and lower loss, as well as research on the different
parameters to adapt.

---

## 2. Hardware and Software Set-Up
The hardware components chosen for this project design are as follows:
* NVIDIA Jetson Nano
* Raspberry Pi 4 module v2 camera
These two main components were used to build the physical system. The Jetson Nano
needs the camera for input, while the image processing and UART communication are
programmed on the Jetson Nano, completing the system.

The components needed to set up the physical hardware of the Jetson Nano was based
on the following link
["Getting Started with Jetson Nano Developer Kit"](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro),
and specific components that we used are listed below:
* monitor with displayPort or HDMI cable, keyboard and mouse (at least for initial setup)
* micro-SD card (at least 32 GB)
* micro-USB power cable or DC-barrel jack power cable
* ethernet cable (or WiFi USB adapter)

To set up the Jetson Nano for deep learning, the following links were used:
* ["Getting Started with the NVIDIA Jetson Nano"](https://www.pyimagesearch.com/2019/05/06/getting-started-with-the-nvidia-jetson-nano/)
* ["How to configure your NVIDIA Jetson Nano for Computer Vision and Deep Learning"](https://www.pyimagesearch.com/2020/03/25/how-to-configure-your-nvidia-jetson-nano-for-computer-vision-and-deep-learning/)

---

## 3. Design
The design of this project is based on the goals and the design requirements
for the system and specifications of the problem. To meet the goals and
requirements of the project the system had to be able to take in a real-time
camera stream (placed on a drone or any moving device), process the images to
trigger when potential crash objects would be detected, and inform a flight
controller of the objects’ location in frame, speed and distance. In order to
design and create such a system, the following software and hardware
specifications were required: An image processing technique implemented in
real-time with high accuracy through the use of neural networks; a small high
speed digital camera with high resolution compatible with the chosen SOM; and
an SOM that is equipped with a camera input port, ethernet port, portable power
supply attachment port, and has high speed processing and large (or expansive)
memory. An SOM that is compatible to communicate with the BrainFPV drone’s
microcontroller and the software it uses must be open source.

This project’s image processing is based on deep learning neural networks to
predict potential collisions with images from real-time camera inputs. Due to
the various environments that drones are used in, there are no particular
objects to train a neural network with, which makes the use of a CNN model alone
is insufficient for the purpose of this project. As a result, an optical flow
algorithm was implemented on the frames, making the velocity vector field images
the dataset for the neural network’s training model. The expected result of
training the machine with dataset of frames processed with optical flow is that
the system can learn to detect potential crashes with the information provided
from vector field images. This section discusses the design and development of
the image processing pipeline.

The image processing software implemented on the Jetson Nano was developed to
calculate the velocity vector field between frames using an OpenCV algorithm for
the Lucas-Kanade optical flow method. For the purpose of this project, the
algorithm was incorporated in several stages of the design process. Applying the
optical flow algorithm to operate in real-time with a camera as the input source
pertains to this project because the overall system calls for a real-time camera
input. Calculations completed by the algorithm acts as post-processing of the
camera capture, extracting information that can not be described from raw images
and the sliding window technique was used in conjunction with the optical. The
sliding windows technique collects a given number of the most recent frames, and
applies optical flow on those frames to calculate the velocity of the pixel
movements between the frames. The use of a sliding window technique allows for
the combination of multiple optical flow calculations, making each output image
a tracking of vector fields throughout several frames, rather than continuously
stacking the previous result from the newly calculated two frames.

As shown in Figure 4, the vector field results for two frames are not
continuously being stacked for all frames captured from start to finish by the
system. Rather it is doing the processing two frames at a time and stacking the
vector field results with results of the next consecutive pair of frames and so
on until optical flow is calculated on 5 frames total and that result is a
single processed frame. The optical flow images for the training model require
preloaded videos as the input. The videos were collected from online drone
footage and saved as "jpeg" images to utilize as the dataset. This use of the
algorithm allows for the CNN to analyze information that is not attained from
the raw camera captures to predict when a crash event will soon occur.

The dataset for this project was generated using Keras’ image data generator.
The dataset was designed to have two label classes: crash and no crash. Images
were organized to make up the training and testing datasets of velocity vector
field images collected from the output of the optical flow algorithm. The images
for training and testing were modified to gray-scale and pixels were rescaled to
accelerate the training of the model. When selecting frames for the dataset, the
images in the testing and training sets must be different from each other. The
more images in a dataset, the larger the neural network, which increases its
accuracy as a result of the improvement in the network’s learning. To increase
the dataset, the images were augmented.

An important part of the training model is the validation dataset, which is
collected from the training datasets and is not trained when compiling the
model. Instead, the validation set is used to evaluate the model’s fitting
through accuracy and loss plots once it is trained. Then adjust the dataset for
both, over-fitting and under-fitting training models, by including more images
to the dataset to account for the model’s accuracy in learning.

The CNN model was trained with velocity vector field images. Keras provides a
sequential CNN model designed with 3 convolutional blocks for learning and one
classification block classifying. After compiling the model, the system was
ready to be trained for learning, validating for evaluation, and testing for
predictions. As shown in Figure 5, the model was trained and tested on a
significant number of epochs, number of iterations the model trains the entire
training dataset. Increasing the epochs increases the machine’s capability of
generalizing the data to adapt to new information.

From evaluations of the model’s accuracy and loss for different numbers of
epochs the model had been shown to be most effective with a minimum of 50 epochs
completed. Since accuracy and loss are two significant factors for the system’s
success, an adequate training dataset size is required to prevent any
over-fitting or under-fitting of the neural network's learning capabilities. To
overcome such problems for the neural network, the training dataset was
increased by including new images from different footage and applying
augmentation, horizontal and vertical flip, a zoom range from 0.8 to 1.2, and a
rotation range from -40 degrees to 40 degrees. These parameters affect the
training model in accuracy, speed, and fitting of the model from training. The
specific values and characteristics of the model are shown in Table 1. This
increase in training dataset size from new images and augmentation increases the
generalization of the machine to recognize features from new data. The results
from providing the machine with a validation set provided information on the
network’s performance. A validation set was generated by withholding 5% of
images from the training dataset and using them to evaluate the model.

---

## 4. Files Description

### [SWOF_functions_complete.py](https://github.com/safwali11/iSpy/blob/master/SWOF_functions_complete.py)


### [SWOF_dataset_build.py](https://github.com/safwali11/iSpy/blob/master/SWOF_dataset_build.py)


### [SWOF_CNN_TRAINING.py](https://github.com/safwali11/iSpy/blob/master/SWOF_CNN_training.py)


### [SWOF_RealTime_PredictionsTesting.ipynp](https://github.com/safwali11/iSpy/blob/master/SWOF_RealTime_PredictionsTesting.ipynb)


### [SWOF_RealTime_TestingWithImageDisplay.py](https://github.com/safwali11/iSpy/blob/master/SWOF_RealTime_TestingWithImageDisplay.py)


### [SWOF_RealTime_withUART.py](https://github.com/safwali11/iSpy/blob/master/SWOF_RealTime_withUART.py)


---

## 5. Contributors
Safwa Ali, Huda Irshad, Daniel Haehn
UMass Boston
Sept 2019 - May 2020
