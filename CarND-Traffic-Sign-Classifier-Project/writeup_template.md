# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./my_images/sample_sign.jpg "Sample Sign"
[image2]: ./my_images/sample_sign_gray.jpg "Grayscaled Sample Sign"
[image3]: ./my_images/hist_train.jpg "Histogram Train Dataset"
[image4]: ./my_images/hist_valid.jpg "Histogram Valid Dataset"
[image5]: ./my_images/hist_test.jpg "Histogram Test Dataset"
[image6]: ./my_images/ConvNet.png "My signals"
[image7]: ./my_images/my_signals.jpg "My signals"

[image8]: ./examples/placeholder.png "Traffic Sign 3"
[image9]: ./examples/placeholder.png "Traffic Sign 4"
[image10]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/avikjuneja/Udacity-SelfDrivingCar_ND-T1/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library and list APIs to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of validation examples = 4410
Number of testing data sets = 12630
Shape of a traffic sign data = (32, 32, 3)
Number of unique classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram chart showing how the data is distributed across classes for Training, Validation and Test Data...

It is obvious from the data set that the distribution is skewed with limited samples for class ids >= 13. This could result in significant inaccuracies when testing against signs in that range.

Training Dataset Distribution
![alt text][image3]

Validation Dataset Distribution
![alt text][image4]

Test Dataset Distribution
![alt text][image5]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

##### a. Gray scale conversion
As a first step, I decided to convert the images to grayscale because it would improve perfomance. The performance improvement could be attributed to reduction in number of channels in the original image from 3 to 1.

Here is an example of a traffic sign image before and after grayscaling.
Original Image
![alt text][image1]

Grayscale converted Image
![alt text][image2]

It should be noted that the improvement in performance was not noticable and conversion to grayscale was dropped due to some accuracy loss.

##### b. Data Normalization

One key component of preprocessing was normalization of the images. Normalization allows bounding the pixel dataset values within range of interest to avoid possible over/underflow during computation. There are various techniques available for normalization. THe following were tested and the first one resulted in maximum accuracy.

    1. Standard Score
        x = (x - x.mean())/x.std()
    2. Feature Scaling
        x = (x-128)/128
    3. Histogram Equalization
        x = histeq(x) // see code for implemenation
    4. Max Normalization
        x = x/ 255.0
    5. Feature Scaling
        minimum = 0
        maximum = 255
        x = a + ((x - minimum)/ (maximum - minimum)


##### c. Additional Data Generation

    1. There are ways to generate additional data for a uniform distribution of training set. This ensures the trained model is not skewed and any bias is avoided in the trained weights. There are various image augmentation techniques that can be employed for this purposes. Due to time limitations, these are reserved for future work and will not be part of this submission. Some posiible data augmentation techniques are:
    i. Flip
    ii. Rotate
    iii. Scaling/Zooming
    iv. ZCA Whitening
    v. etc..


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride,'valid' padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5     	| 1x1 stride,'valid' padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling (conv2)  	| 2x2 stride,  outputs 5x5x32, output 800		|
| Max pooling (conv1) 	| 2x2 stride,  outputs 14x14x16 output 784		|
| Fully connected		| Concat (flat(conv1), flat(conv2)) out: 1584	|
| Fully connected		| output: 400  									|
| Fully connected		| output: 43  									|
| Drouput				| output:43    									|
| Softmax				| etc.        									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the follwing:

    1. Adam Optimizer
    2. softmax_cross_entropy_with_logits
    3. Batch size = 50
    5. Epochs = 25
    6. Hyperparameters:
        mu = 0
        sigma = 0.1
        Training Rate = 0.0005
    
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
Train Accuracy = 1.000
Valid Accuracy = 0.974
Test Accuracy = 0.960

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

First architecture was inspired by the LeNet Lab and used as a starting point. It was chosen due to prior experience and ease of implementation.

* What were some problems with the initial architecture?

The first model was optimized for digit recognition and possibly for single channel inputs. The accuracy of the LeNet Architecture was limited to 89-93%

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

LeNet Archtiecture:

1. Initially the original LeNet model was adjusted to support 3-channel inputs for RGB images and output feature size of 43 (unique labels), and compared against 1-channel grayscaled images with 43 feature output (unique labels). This resulted in minimal validation accuracy improvement. 
2. Then the feature size was varied for each layer in order to extract more features and maximize accuracy. Going from feature size set of {3->6->16->400->120->84->43} to {3->16->32->800->120->43} did not have any noticeable impact on the accuracy.

3. Further, I tried a ConvNet based architecture proposed by [Pierre and Yann](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), which resulted in significant improvements in the accuracy.
It is a 2-stage ConvNet architecture with 2-levels of 'convolution and subsamplings'. It is quite similar to the LeNet arch, with a difference in the last classifier/fully connected stage. The Classifier stage consists of: 
    1. Flattened output of 2nd Stage
    2. Flattened output of 1st stage after additional pooling
    3. Concatenation of above two Flattened outputs
4. In order to improve the accuracy futher, an additional fully connected stage with fewer features was added.
5. Adding a 'Dropout' Layer further improved the accuracy of the validation and test cases.

The ConvNet based architecture is depicted in the image below:

![alt text][image6]
 

* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The following parameters were tuned with corressponding reasons:
    1. Epoch: This allows convergance to a single accuracy point by iterating over the training set by use of backward propogation and weight adjustment. This also helped tune the sampling rate, as explained later. An appropriate epoch count is required to avoid over fitting the training set (large epoch), vs under training due to small epoch. Large epoch also results in greater performance requirements and longer run times. Epoch of 25 was an ideal value for this architecture. 
    
    2. Batch Size: An appropriate batch size allows training the model with parallel inputs and applying same weights to them. This results in faster training, but can result in lower accuracy. The batch count was varied and settled at 50 for an optimal performance and accuracy.

    3. Sample Rate: Sampple rate allows the rate at which the weights can be adjusted. Large sample rate will allow faster ramp (lesser Epochs) towards steady state, but could result in large steady state error in accuracy. Whereas, smaller sample rate will lead to higher accuracy but only larger Epochs with longer run times.
    
    4. Feature size: Based on the number of output feature size of 43, each layer's output feature size had to be increase. This allowed more information to be parsed in the images with higher accuracy. The increased features came at a cost of performance (longer run times per Epoch) due to higher computational requriements.
    
    5. Dropout Rate: The Dropout rate is reflected in the 'keep_prob' variable, which determines how many signals to ignore/keep when determing the new weights. This avoids overfitting of data. The value for shmooed and 0.3 resulted in maximum validation and test accuracy.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
 Using the ConvNet based architecture with layer one output into the classifier in parallel to the layer 2 output resulted in higher accuracy. This is attributed to attainment of features from upper layers for better classification of images. The parallel input was achieved by concaternating the output of layer 1 with that of layer 2 in the classifier stage.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7]

All images were easily classified.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)	| Speed limit (70km/h) Accuracy=100%			| 
| No Passing			| No Passing Accuracy=100%						| 
| No Passing			| No Passing Accuracy=100%						| 
| Road Work				| Road Work Accuracy=100%						| 
| Stop					| Stop Accuracy=100%							| 
| Priority Road			| Priority Road Accuracy=100%					| 


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.6%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located towards the 2nd to last cell of the Ipython notebook.

For the first image, the model is very sure that this is a 'Speed limit (70km/h)' sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (70km/h)							| 
| .00     				| Speed limit (30km/h)							|
| .00     				| Speed limit (20km/h)							|
| .00     				| Speed limit (50km/h)							|
| .00     				| General Caution								|


For the second image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No passing   									| 
| .00     				| End of no passing								|
| .00					| Vehicles over 3.5 metric tons prohibited		|
| .00	      			| No passing for vehicles over 3.5 metric tons	|
| .00				    | Speed limit (20km/h) 							|


For the third image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No passing   									|
| .00	      			| No passing for vehicles over 3.5 metric tons	|
| .00					| Vehicles over 3.5 metric tons prohibited		|
| .00				    | Speed limit (80km/h) 							|
| .00     				| Dangerous curve to the left					|

For the fourth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Road work   									|
| .00	      			| Beware of ice/snow	 						|
| .00					| Wild animals crossing 						|
| .00				    | Bumpy road 									|
| .00     				| Right-of-way at the next intersection			|


For the fifth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop											| 
| .00     				| Speed limit (80km/h)							|
| .00     				| Speed limit (30km/h)							|
| .00     				| Speed limit (50km/h)							|
| .00     				| Speed limit (60km/h)							|


For the sixth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority road									|
| .00	      			| End of all speed and passing limits			|
| .00					| Right-of-way at the next intersection			|
| .00     				| End of no passing by vehicles over 3.5 metric tons|
| .00				    | Speed limit (80km/h) 							|

