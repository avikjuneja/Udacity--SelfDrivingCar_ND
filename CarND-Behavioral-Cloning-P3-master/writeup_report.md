# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nv_network.png "Model Visualization"
[image2]: ./examples/center_lap.jpg "Center Lap"
[image3]: ./examples/recover_1_0.jpg "Recovery Image"
[image4]: ./examples/recover_1_1.jpg "Recovery Image"
[image5]: ./examples/recover_1_2.jpg "Recovery Image"
[image6]: ./examples/recover_1_3.jpg "Recovery Image"
[image7]: ./examples/recover_1_4.jpg "Recovery Image"

[image8]: ./examples/recover_2_0.jpg "Recovery Image"
[image9]: ./examples/recover_2_1.jpg "Recovery Image"
[image10]: ./examples/recover_2_2.jpg "Recovery Image"
[image11]: ./examples/recover_2_3.jpg "Recovery Image"

[image13]: ./examples/center_lap.png "Normal Image"
[image14]: ./examples/center_flipped.jpg "Flipped Image"
[image15]: ./examples/center_cropped.jpg "Cropped"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 recoding of driving in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes, and depths between 1 and 100 (model.py lines 118-1) 

The model includes RELU layers to introduce nonlinearity (code lines 118-122), and the data is normalized in the model using a Keras lambda layer (code line 116). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines ). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 100-106). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 130).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used:
1. A combination of center lane driving, recovering from the left and right sides of the road.
2. Recovery driving trying to mimic path correction when car was about to go off track
3. Data collected by driving in opposite driving direction to mimic right hand road curves

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to have enough convolution layers for various driving and recover characteristics. 

My first step was to use a convolution neural network model similar to the NVIDIA autonomous vehicle network architecture. I thought this model might be appropriate because it provides lowest training and validation loss along with smoothest driving experience, as compared with simple LeNet based and other architectures.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Due to vaious driving conditions collects, the training and validation had very low mean squared error. The validation loss was equal to to lower than training loss. This assured the model was not overfitting. 

Including Dropout layer didn't really help improve the model, as the validation and training loss already had very low and almost equal mean square error loss. But it was included to avoid overfitting anyway.

Converting the training and validation images from BGR (original format) to RGB improved inferencing significantly. This is because the input data into the model was indeed in BGR format.

Cropping the images to remove the background and hood of the car helped further reduce training and validation loss. The cropped image looks like:

![alt text][image13]
![alt text][image15]

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I added training data with 'recovery' driving at those curves, along with driving in reverse direction.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes .\

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))  ## normalization step
model.add(Cropping2D(cropping=((50,20),(0,0))))    ## crop image to clip background scenery and hood of the car to focus on road
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))   ## output steering angle

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

Right Side Recovery:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]

Left side recovery:
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]

To augment the data sat, I also flipped images and angles thinking that this would mimic driving on opposite curves. For example, here is an image that has then been flipped:

![alt text][image13]
![alt text][image14]


After the collection process, I had X number of data points. I then preprocessed this data by normalizing using the following formula:

x/255.0 - 0.5


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by saturating validation and training loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
