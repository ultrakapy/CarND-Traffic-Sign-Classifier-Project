# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: Test1.jpg "Traffic Sign 1"
[image5]: Test2.jpg "Traffic Sign 2"
[image6]: Test3.jpg "Traffic Sign 3"
[image7]: Test4.jpg "Traffic Sign 4"
[image8]: Test5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/ultrakapy/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the numpy library and basic Python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

For an exploratory visualization of the dataset, please refer to the Jupyter notebook for the project. I wrote some code that randomly chooses an image from the training set as it was recommended to start with something simple.


### Design and Test a Model Architecture

For preprocessing, I simply normalized the data using the formula: (pixel - 128) / 128.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5 filter     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding,  outputs 14x14x6 				|
| Convolution 5x5 filter	    | 1x1 stride, valid padding, outputs 10x10x16 |
|RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding,  outputs 5x5x16 				|
| Flatten					| |
| Fully connected		| Output 120        		|
| RELU					| |
| Fully connected		| Output 84        		|
| RELU					| |
| Fully connected		| Output 10       		|


To train the model, I used an AdamOptimizer with batch size of 32, a learning rate of 0.001, and run ran the model for 25 epochs.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.939
* test set accuracy of 0.931

The model was based on the LeNet architecture. Out of the box, this architecture was able to achieve a validation accuracy of about 0.85. After normalizing the data, tuning the batch size and the number of epochs, the model was able to achieve a validation accuracy of more than 0.93.
 

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The 2nd, 4th and 5th images might be difficult to classify because they are similar in that they are all triangles with slightly different images in the middle and the colors are also the same. For example, even though the 5th image of "bumpy road" was misclassified by my model, the class it predicted was of another triangle sign.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 17 - No Entry      		| 17 - No Entry   									| 
| 28 - Children Crossing     	| 8 - Speed limit (120km/h) 										|
| 1  - Speed limit (30km/h)	|5 - Speed limit (80km/h) 											|
| 25 - Road work	      		| 25 - Road work					 				|
| 22 - Bumpy road		| 12 - Priority road      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This does not compare favorably to the accuracy of the test set. I believe the reason is because some of the images I found online had "watermarks" in them. Another reason was a combination of the quality of images when resized to 32x32 and the model only having a 93% test accuracy. Unfortunately I did not have a lot of time to tune the model due to the upcoming course deadline so once I was able to get the validation past the minimum accuracy required I moved on so I could have enough time to complete the project and the remaining course projects.

For the first and fourth images, the model is very sure with probability of 1.0 in both cases. And for the third and fifth, the models' prediction is 0.0. I was a bit surprised by the latter, particularly the 3rd image (Speed limit (30km/h)) and was predicted to be another speed limit sign but with 80km/h instead. I was expecting the probabilities to be closer since they are so similar. The probability for the second image was also quite surprising but then again there are quite a lot of signs that are triangles and in this case the second hightest probability sign is "Right-of-way at the next intersection" (class Id 11) which has probability 42% and happens to be a triangle sign.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| 17 - No Entry   									| 
| .03     				| 28 - Children Crossing  										|
| .00					| 1  - Speed limit (30km/h)											|
| 1.00	      			| 25 - Road work					 				|
| .00				    | 22 - Bumpy road     							|





