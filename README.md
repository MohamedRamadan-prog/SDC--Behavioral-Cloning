# SDC--Behavioral-Cloning
It is a supervised regression algorithm between the car steering angles and the road images in front of a car. Those images were taken from three different camera angles (from the center, the left and the right of the car). - The network is based on The NVIDIA model, which has been proven to work in this problem domain. - As image processing is involved, the model is using convolutional layers for automated feature engineering.

# Module Algorithm:
We collected the dataset, which included; images of the road periodically and their corresponding steering angle, Then we used the dataset to train the NN so it predicts the correct steering angle corresponding to the current input image from the camera.
- The following block diagram shows the steps of the algorithm.

![image](https://user-images.githubusercontent.com/53750465/62554751-fd2e5c80-b871-11e9-8a74-05cb0f270bef.png)


# Building a Neural Network
- We used NVIDIA Neural Network, whose architecture consists of a normalization layer, is followed by five convolutional layers, then followed by five fully-connected layers, finally end a single output node as shown at the following figure.

![image](https://user-images.githubusercontent.com/53750465/62554810-1a632b00-b872-11e9-8cbc-f5119b4fe7c2.png)


# Collecting training & validation data
We used Udacity simulator to collect training and validation dataset. The simulator supports two modes: Training mode for training and autonomous mode for testing.
- Training mode enables you to drive the car manually so the simulator collects and sorts the steering angles with their corresponding images of the road, then the simulator outputs:
I) IMG folder: this folder contains all the frames of your driving.
II) Driving_log.csv: this file contains each row in this sheet correlates your image with the steering angle, throttle, brake, and speed of your car as shown in the following figure.


![image](https://user-images.githubusercontent.com/53750465/62554939-572f2200-b872-11e9-89af-89f40f79df89.png)


# Neural Network Training
- Before training the neural network there are four preprocessing steps:
I) Reading & storing the training data:
- We used python CVS library to read & store the lines from "Driving_Log_csv" file, that we get from the simulator, then for each line we extract the path to the camera image and load the image by using OpenCV finally append it to a list of images.
- Doing the same to extract the steering measurements "output label".
II) Integrating images of the three cameras
- As the simulator captures images from three cameras mounted on the car (center, right and left camera), we took advantage of using multiple cameras to increase our data set 3 times than using only center camera.


![1](https://user-images.githubusercontent.com/53750465/62555084-965d7300-b872-11e9-8f91-5e7ac3e921c4.JPG)


# Validatation of the Network
- After training the model, validation phase is necessary to identify if the model performance is acceptable or not.
- We looked forward to balanced case so we observed the Mean Squared Error behavior in training and validation phases.
- When the model predictions are poor on both the training and validation set as the Mean Squared Error high on both, then this is evidence of under-fitting So there are possible solutions could be to
1) Increasing the number of epochs
2) Adding more convolutions to the network.
- When the model predicts well on the training set but poorly on the validation set as low mean squared error for training set, high mean squared error for validation set, this is an evidence of over-fitting, so the solution is:
I) using dropout or pooling layers.
II) using fewer convolution or fewer fully connected layers.
III) collecting more data or further augment the data set
- we used 2 epochs during the validation, the MSE decreases within 2 epochs then starts to increases again as shown in the following figur


![image](https://user-images.githubusercontent.com/53750465/62555180-c442b780-b872-11e9-9552-27109db9fd32.png)

# Network Testing
- To Test the Neural Network, we used the autonomous mode of the simulator.
- The simulator displays current angle that the network predicts.


![image](https://user-images.githubusercontent.com/53750465/62555254-e76d6700-b872-11e9-9c2d-4b7509642ee6.png)

