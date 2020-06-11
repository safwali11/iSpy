# iSpy
Real Time Collision Avoidance Deep Learning Image Processing Alert System

INCLUDE PROJECT PROCEEDINGS PDF


## Table of Contents


## Introduction
This project presents a real-time collision avoidance alert system using deep learning image processing to trigger a microcontroller. The motivation for this project is to integrate the system with first-person-view drone flight controllers to assist drone pilots in avoiding potential collisions. To predict potential collisions in real time, the system inputs real-time velocity vector field calculations into a convolutional neural network, developed and trained on the NVIDIA™ Jetson Nano, a small artificial intelligence computer. A convolutional neural network is a class of deep learning used for image recognition and classification by applying convolution filter layers to learn features in a dataset for future predictions. Vector field calculations are performed by way of optical flow which is the process of detecting and calculating the movement of pixels between consecutive frames. A threshold was set to trigger an alert to the drone flight controller through a universal asynchronous receiver/transmitter, a device for data transmission. The training model achieved 85.6\% accuracy with 33.6\% loss. When the system was tested, it was able to predict and alert with approximately 72\%.

The image processing components of this project were merged into one system and implemented on the Jetson Nano’s GPU. Figure 3 shows the overall image processing pipeline. A real-time video stream is analyzed through an optical flow algorithm to generate images of the vector field between the frames. The vector field images are then analyzed through CNN to predict if the images are crash or no crash images. As shown in Figure 4, to create the complete CNN, pre-recorded drone frames processed through an optical flow algorithm were used to train a model. To determine if the drone is about to crash onto any objects within the frame, the CNN prediction values are compared to the potential crash threshold value. If there are potential crash objects in the frame, an alert is communicated to the drone flight controller through UART.

Since the training model is predicting crash or no crash in the frame, a boundary has to be determined and set to alert the system before we encounter a frame detected as crash. The solution is using the training model’s prediction values of crash and no crash per image to understand the danger level of a crash occurring. When the prediction for crash is  the threshold value of 0.3 (meaning prediction of crash is 30\% and the frame is labeled as no crash), the SOM is required to convey the potential crash to the main flight controller. The 30\% threshold shows that the camera is not detecting a crash yet, but there are enough features in the image that are similar to that of a crash. Although it has not been tested, with that threshold value, the goal is that the flight controller has enough time to respond to the alert.

As shown in Figure 6, when a potential crash object is detected based on the comparison to the threshold value, a signal is communicated. For testing, a phrase was printed as the communication signal. The code for UART was written to transmit the signal that is decoded as “crash” and then close immediately. UART is the final form of communicating the signal.

There is still continuous progress on this project to improve the training model for higher accuracy and lower loss, as well as research on the different parameters of the system.

## Set Up
The hardware components chosen for this project design are the NVIDIA Jetson Nano as the SOM and the Raspberry Pi 4 module v2 camera for the real-time camera stream input. These two main components were used to build the physical system. The Jetson Nano needs the camera for input, while the image processing and UART communication are programmed on the Jetson Nano, completing the system.

The Jetson Nano is an SOM that can run high speed modern AI algorithms, making it a small AI computer. It contains connectors and ports for ethernet, microSD card, HDMI output, DisplayPort, DC barrel jack 5V power input, USB, and MIPI CSI camera. The advantage of this SOM is that it can run multiple neural networks in parallel and process several high-resolution sensors simultaneously, which makes it ideal for computer vision and high performance computing. In order to use the Jetson Nano for deep learning image processing, the environment was first assembled with NVIDIA’s JetPack and essential packages and libraries for computer vision. Python allows for many open-source libraries to be incorporated, such as OpenCV (which has many resources for real-time applications). TensorFlow and Keras were installed. Jupyter Notebook was also installed and used to simulate the algorithms and neural networks before implementing the processing on the Jetson Nano’s GPU. 

The Raspberry Pi module v2 camera is a high resolution camera (3280 x 2464 pixels) that can capture approximately 90 frames per second and is compact in size and compatible with the Jetson Nano. It connects through a ribbon cable to a CSI port, which makes it possible to connect to the Jetson Nano’s MIPI CSI camera connector port. The Raspberry Pi camera was connected to the Jetson Nano and real-time stream was ensured, using OpenCV source code, which was later used for iterative testing of the image processing codes.


## Design
The design of this project is based on the goals and the design requirements for the system and specifications of the problem. To meet the goals and requirements of the project the system had to be able to take in a real-time camera stream (placed on a drone or any moving device), process the images to trigger when potential crash objects would be detected, and inform a flight controller of the objects’ location in frame, speed and distance. In order to design and create such a system, the following software and hardware specifications were required: An image processing technique implemented in real-time with high accuracy through the use of neural networks; a small high speed digital camera with high resolution compatible with the chosen SOM; and an SOM that is equipped with a camera input port, ethernet port, portable power supply attachment port, and has high speed processing and large (or expansive) memory. An SOM that is compatible to communicate with the BrainFPV drone’s microcontroller and the software it uses must be open source.

This project’s image processing is based on deep learning neural networks to predict potential collisions with images from real-time camera inputs. Due to the various environments that drones are used in, there are no particular objects to train a neural network with, which makes the use of a CNN model alone is insufficient for the purpose of this project. As a result, an optical flow algorithm was implemented on the frames, making the velocity vector field images the dataset for the neural network’s training model. The expected result of training the machine with dataset of frames processed with optical flow is that the system can learn to detect potential crashes with the information provided from vector field images. This section discusses the design and development of the image processing pipeline.

The image processing software implemented on the Jetson Nano was developed to calculate the velocity vector field between frames using an OpenCV algorithm for the Lucas-Kanade optical flow method. For the purpose of this project, the algorithm was incorporated in several stages of the design process. Applying the optical flow algorithm to operate in real-time with a camera as the input source pertains to this project because the overall system calls for a real-time camera input. Calculations completed by the algorithm acts as post-processing of the camera capture, extracting information that can not be described from raw images and the sliding window technique was used in conjunction with the optical. The sliding windows technique collects a given number of the most recent frames, and applies optical flow on those frames to calculate the velocity of the pixel movements between the frames. The use of a sliding window technique allows for the combination of multiple optical flow calculations, making each output image a tracking of vector fields throughout several frames, rather than continuously stacking the previous result from the newly calculated two frames.
 	
As shown in Figure 4, the vector field results for two frames are not continuously being stacked for all frames captured from start to finish by the system. Rather it is doing the processing two frames at a time and stacking the vector field results with results of the next consecutive pair of frames and so on until optical flow is calculated on 5 frames total and that result is a single processed frame. The optical flow images for the training model require preloaded videos as the input. The videos were collected from online drone footage and saved as "jpeg" images to utilize as the dataset. This use of the algorithm allows for the CNN to analyze information that is not attained from the raw camera captures to predict when a crash event will soon occur.

The dataset for this project was generated using Keras’ image data generator. The dataset was designed to have two label classes: crash and no crash. Images were organized to make up the training and testing datasets of velocity vector field images collected from the output of the optical flow algorithm. The images for training and testing were modified to gray-scale and pixels were rescaled to accelerate the training of the model. When selecting frames for the dataset, the images in the testing and training sets must be different from each other. The more images in a dataset, the larger the neural network, which increases its accuracy as a result of the improvement in the network’s learning. To increase the dataset, the images were augmented.
	 
An important part of the training model is the validation dataset, which is collected from the training datasets and is not trained when compiling the model. Instead, the validation set is used to evaluate the model’s fitting through accuracy and loss plots once it is trained. Then adjust the dataset for both, over-fitting and under-fitting training models, by including more images to the dataset to account for the model’s accuracy in learning. 

The CNN model was trained with velocity vector field images. Keras provides a sequential CNN model designed with 3 convolutional blocks for learning and one classification block classifying. After compiling the model, the system was ready to be trained for learning, validating for evaluation, and testing for predictions. As shown in Figure 5, the model was trained and tested on a significant number of epochs, number of iterations the model trains the entire training dataset. Increasing the epochs increases the machine’s capability of generalizing the data to adapt to new information.

From evaluations of the model’s accuracy and loss for different numbers of epochs the model had been shown to be most effective with a minimum of 50 epochs completed. Since accuracy and loss are two significant factors for the system’s success, an adequate training dataset size is required to prevent any over-fitting or under-fitting of the neural network's learning capabilities. To overcome such problems for the neural network, the training dataset was increased by including new images from different footage and applying augmentation, horizontal and vertical flip, a zoom range from 0.8 to 1.2, and a rotation range from -40 degrees to 40 degrees. These parameters affect the training model in accuracy, speed, and fitting of the model from training. The specific values and characteristics of the model are shown in Table 1. This increase in training dataset size from new images and augmentation increases the generalization of the machine to recognize features from new data. The results from providing the machine with a validation set provided information on the network’s performance. A validation set was generated by withholding 5% of images from the training dataset and using them to evaluate the model.


## Files

### SWOF_functions_complete.py


### SWOF_dataset_build.py


### SWOF_CNN_TRAINING.py


### SWOF_RealTime_PredictionsTesting.ipynp


### SWOF_RealTime_TestingWithImageDisplay.py


### SWOF_RealTime_withUART.py

