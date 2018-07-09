## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/calibration_input_1.jpg "Calibration Input"
[image1]: ./output_images/undistorted_input_1.jpg "Undistorted"
[image2]: ./output_images/report_images/original.jpg "Original"
[image3]: ./output_images/report_images/undist.jpg "Undistorted"
[image4]: ./output_images/report_images/gray_undist.jpg "Gray Undist"
[image5]: ./output_images/report_images/color_binary.jpg "Color Binary"
[image6]: ./output_images/report_images/grad_binary.jpg "Grad Binary"
[image7]: ./output_images/report_images/combined_binary.jpg "Combined Binary"
[image8]: ./output_images/report_images/color_warped.jpg "Color Warped"
[image9]: ./output_images/report_images/binary_warped.jpg "Binary Warped"
[image10]: ./output_images/report_images/binary_src.jpg "Binary with Src points"
[image11]: ./output_images/report_images/histogram.jpg "Histogram"
[image12]: ./output_images/report_images/result.jpg "Result"
[image13]: ./output_images/report_images/lane_windows.jpg "Lane Windows"

[image14]: ./output_images/roi.jpg "Region of Interest"
[image15]: ./output_images/combined_thresh_straight_lines1.jpg "Warped"
[image16]: ./output_images/color_thresh_test2.jpg "Color Binary "
[image17]: ./output_images/grad_thresh_test2.jpg "Gradient Binary "
[image18]: ./output_images/combined_thresh_test2.jpg "Combined Binary "

[image19]: ./output_images/output_corners/corners_calibration2.jpg "Chessboard Corners"

[video1]: ./project_out.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the third code cell of the IPython notebook located in "./CarND-Advanced_Lane_Finding.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Corner Detection:

![alt text][image19]

Original:

![alt text][image0]

Undistorted:

![alt text][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

![alt text][image3]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in code cell 16 at lines #21 through #48 in `"./CarND-Advanced_Lane_Finding.ipynb"`).  Here's an example of my output for this step. 

![alt text][image5]

![alt text][image6]

![alt text][image7]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in in the 8rd code cell of the IPython notebook).  The `warp()` function takes as inputs an image (`img`) and uses an internal source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
#global variables:
src_x = (200,170,605,605)
src_y = (0,440)
dst_x = (350,1280-320)

def warp(image):
    ...
    src = np.float32([[src_x[0],img_size[1]-src_y[0]], [img_size[0]-src_x[1],img_size[1]-src_y[0]], [img_size[0]-src_x[2], src_y[1]], [src_x[3],src_y[1]]]) 
    
    dst = np.float32([[dst_x[0],img_size[1]],[dst_x[1],img_size[1]],[dst_x[1],1],[dst_x[0],1]])
    ...
```

This resulted in the following source and destination points:

a. Includes hood of car:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 350, 720      | 
| 1110, 720     | 960, 720      |
| 675, 440      | 960, 0        |
| 605, 440      | 350, 0        |

b. Excludes hood of car

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 260, 675      | 350, 720      | 
| 1045, 675     | 960, 720      |
| 675, 440      | 960, 0        |
| 605, 440      | 350, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image14]

![alt text][image15]


Also, tried the same src and dst on other images:

![alt text][image10]

![alt text][image8]

![alt text][image9]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image13]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in code cell 14 (lines 1 to 25) and cell 22 at lines #48in my code in the IPython notebook.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image12]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

There are few things that were not taken into account and need to be understood as to how they can be incorporated:

    1. The example output image (./examples/example_output.jpg) shows the lane colored region to include the hood of the car. However, my submission was not accepted with comments to exclude the hood. This was not mentioned in the specification. I am re-submitting to exclude the hood.     
    2. The non-linear shape of the hood of the car could be removed from the warp transform, but that further leads to ignoring some crucial segments of the lane lines. One possible solution would be to include the hood while 'warping' but somehow exclude when drawing the lane area.    
    3. This code hasn't been tested on following scenarios:
        a. An object or another vehile right in front of the car (inside the lane)        
        b. Shorter view of the lanes ahead (smaller than assummed lane depth. One way to tackle this would be to adopt to the depth of the lane visible in front, instead of assumming a fixed field of view for the lanes. This is of challenge particularly in the harder_challenge.mp4        
    4.  cv2.getPerspectiveTransform(src,dst) doesnt seem to accept polygons with more than four points. This is further required to ignore some areas like the hood of the car.    
    5. The code hasn't been tested in other light conditions. e.g. night time, blaring sun with scattered rays.

