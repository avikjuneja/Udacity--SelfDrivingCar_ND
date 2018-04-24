# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[//]: # (Image References)

[image1]: ./test_output/test_images/output_pipe0a_solidYellowCurve.jpg "color_amplify_mask"
[image2]: ./test_output/test_images/output_pipe0b_solidYellowCurve.jpg "color_dark_mask"
[image3]: ./test_output/test_images/output_pipe1a_solidYellowCurve.jpg "Grayscale"
[image4]: ./test_output/test_images/output_pipe1b_solidYellowCurve.jpg "Gray_OR_color_amp"
[image5]: ./test_output/test_images/output_pipe2_solidYellowCurve.jpg "Gaussian_Filter"
[image6]: ./test_output/test_images/output_pipe3_solidYellowCurve.jpg "Canny_out_edges"
[image7]: ./test_output/test_images/output_pipe4_solidYellowCurve.jpg "Canny_with_color_mask"
[image8]: ./test_output/test_images/output_pipe5_solidYellowCurve.jpg "Region_of_interest"
[image9]: ./test_output/test_images/output_pipe6_solidYellowCurve.jpg "Masked Edges"
[image10]: ./test_output/test_images/output_pipe7_solidYellowCurve.jpg "Hough_Trans_line_seg"
[image11]: ./test_output/test_images/output_pipe8_solidYellowCurve.jpg "Hough_Trans_lane"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 8 steps. They are as follows:

Pipe0: First, I created a masks out of the color thresholds. This was an important step for better and accurate edge detection. 
    a. First mask is to amplyfiy the lane colors yellow and white
    b. Second mask was to mask out dark colors seen on roads, due to shadows, patches etc..
    
Pipe1:
    a. In parallel, I converted the original images to grayscale
    b. Then I applied the color_amp mask to grayscale to enhance the lanes
    
Pipe2: The above grayscale image was then filtered using gaussian filer to create a gray_blur

Pipe3: The gray_blur was then processed using the Canny funtion to detect color gradients in the image for edge detection

Pipe4: I then applied the dark_color mask to the edges to remove redundant edges corressponding to dark patches and shadows

Pipe5/6: Then I further narrowed down the edges using a 'region of interest' mask to focus on lane area in the image to create 'masked_edges'

Pipe7: The 'masked_edges' were then transformed using Hough Transform to produce lines with set thresholds. This allowed further filtering of redundant lines and were plotted using 'draw_lines() function.

Pipe8: In order to draw a single line on the left and right lanes, I added an extrapolate_lines() function. This function is defined to perform the following tasks to produce the lines of interest:
    a. Separate out line segments for left and right lanes using slope thresholds
    b. For each lanes, use the line segment end points as an input to polyfit() function, which results in a line with the best fit (slope and intercept)
    c. Use the best fit line equation to determine end points within region of interest
    d. Then I plotted these lane lines over original image to cover the actural lanes


The pipeline steps are depicted using sample images below: 

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen if there are residual lane lines in the middle of the lane

Another shortcoming could be ambient lighting conditions, significant shadows over one lane, missing lines for a short distance etc...

Another could be if the car is disoriented, and the lane lines are out of the region of interest.

Another issue would be if some of the lane lines do not have distorted edges - might throw off hough transform, since the points might not line up within threshold


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to stabilize the lane lines drawn - possibly by sharing data between successive frames in a video and applying some sort of moving average

Another potential improvement could be to instead of looking at edges only, look for thickness of lanes - to futher isloate stray lines
