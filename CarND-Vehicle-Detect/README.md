
# Vehicle Detection Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./hog_prac/orient_1_cell8_pix4.png
[image2]: ./hog_prac/orient_3_cell8_pix4.png
[image3]: ./hog_prac/orient_13_cell8_pix4.png
[image4]: ./stat_data.png
[image5]: ./output_images/2search_windows2.png
[image6]: ./output_images/0search_windows3.png
[image7]: ./output_images/3search_windows2.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in `helper.py` on line 74-85.
I started by reading in all the vehicle and non-vehicle images. I then tried to get an intuition for the kind of HOG feature values I would obtain based on differing parameters. Here are a few images from my exploration:

![alt text][image1]
![alt text][image2]
![alt text][image3]

Other functions that facilitated extraction of the hog features are “extract_features” and “single_image_features” which can be found on lines 89 and 133, respectively, in helper.py. These functions would specify how many channels in a color space would be used for HOG extraction and they standardized the size of each image before extracting features.


#### 2. Explain how you settled on your final choice of HOG parameters.

I settled on my final choice of HOG parameters by see how well a classifier could predict on a validation set based on a specific permutation of color space and HOG parameters. Here are the different trials I ran through to come to my choice.

![alt text][image4]

After deciding on LUV as the color space based on a preliminary test, I adjusted the orientation value and pixels per cell to come to my choice of parameters. I then performed a test to see if increasing cell per block could help my score and I found a slight improvement going to 4 cells per block as opposed to 2. However, I wasn’t pleased by how long the feature extraction was taking when detecting on the final project video. To speed up time I increased the pixels per cell to 16. I then ran cross validation on the training set and found the YUV space to work the best for extracting the HOG features.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a Linear SVM with c value=0.001 to minimize false positives.  Although other SVM kernels performed better on testing the Linear SVM is much quicker so I used it for the final detection process.
The actual implementation can be found in pipeline.py on lines 81-96 in the function trainModel which takes a training and test set, normalizes the training and test to a normal gaussian, and then fits the data set to the SVM chosen. The function will then predict on the test set and provide a score for the accuracy of its performance. 


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented window search on lines 175-213 and on lines 218-244 in the file helper.py. I decided upon my final window sliding area by playing with different parameters for the overlap and area of the window slide. The windows cover the bottom right of the image because the project video is only concerned with detecting cars in that area. The different square windows I use have side lengths of 64, 80, and 96. They overlap in such a way that one area of the window slide zone is overrepresented in the heatmap. If the heatmap is biased, then it would be hard to choose an appropriate cutoff threshold for the heatmap.

![alt text][image5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To optimize the performance of the classifier I removed the KITTI_extracted dataset because it was causing issues with detecting the white car in the project video at long distances. To minimize false positives, I sampled the project video, hard mining for false positives. I then included these false positives in the non-vehicle dataset used for training. The classifier was also optimized by tuning the parameters in my feature extraction function discussed earlier. I also discovered how each window was performing by saving the windows detected for each scale and overlaying them on the test images.
Here are some example images:

![alt text][image5]
![alt text][image6]
![alt text][image7]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. This process is handled in lines 12-28 of cell 4 in jupyter notebook final.ipynb. I constructed bounding boxes to cover the area of each blob detected. If the blobs had an area less than 2000 I would remove them. This is done in the function draw_labeled_bboxes in helper.py. The bounding boxes are only placed after every 10 frames to increase stability and further defend against false positives. Each bounding box overlaid on the video was an average of the previous 10 frames. This averaging was done in the Boxes class defined in cell 6 of the jupyter notebook final.ipynb.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest issues I faced were from the choice of windows for the window slide and the parameters set for the feature extraction, especially for HOG features. I went through many different iterations of window slide areas. Some windows were much smaller but ended up giving me more false positives, so I didn’t include them in the final pipeline. My pipeline performs pretty well on the video with only one false positive, but due to this false positive I could see the video having more difficulty on a street with more external activity like in a city. My pipeline also takes advantage of the project video constraint that the cars being detected are in the bottom right corner. I would be curious to see how the pipeline perform if the car changed lanes. My pipeline will also fail if the relative speed of the cars in view is much different than our car. Because I average 10 frames to make my boxes it would likely miss the car if it changes positions in the video too quickly.
To make the pipeline more robust I would increase the area of the window slides to cover more of the camera view. I would also create a queue that takes in a car class object so I could average the frames in that way. This could help me have more concentrated and distinct boxes for the car and save information pertaining to a specific car. This could slow down the pipeline though. 


