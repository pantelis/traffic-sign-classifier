# **Traffic Sign Classifier**

Pantelis Monogioudis

NOKIA

## Summary
For German traffic signs classification, the Keras API was used to define a deep neural network and train its hyperparameters. For optimizing the hyperparameters of the model, the scikit-learn python library search API was used as described in the model training section. We have investigated various architectures starting from the LeNet network and adding overfitting mitigation and complexity reduction layers. The performance of the test dataset as well as the performance of random 5 images collected from the internet shows that the selected architecture as well as the hyperparameter search (although not exhaustive) performs well and can be easily replicated by cloning two separate repos: the [project code](https://github.com/pantelis/traffic-sign-classifier) and the [project data](https://github.com/pantelis/traffic-signs-data).  The goals / steps of this project are the following:
* Load, Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

[//]: # (Image References)

[dataset-summary]: ./traffic-signs-dataset-statistics.png "Dataset Summary"
[dataset-visual-histogram]: ./traffic-signs-visual-histogram.png "Visual Histogram"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[dataset-summary]: ./traffic-signs-dataset-statistics.png "Dataset Summary"
[dataset-visual-histogram]: ./traffic-signs-visual-histogram.png "Visual Histogram"
[nested-cv]: ./nested-cv.jpg "Nested CV"
[simple-cv]: ./simple-cv.jpg "Simple CV"
[sign1]: ../traffic-signs-data/sign1.jpg "Traffic Sign 1"
[sign2]: ../traffic-signs-data/sign2.jpg "Traffic Sign 2"
[sign3]: ../traffic-signs-data/sign3.jpg "Traffic Sign 3"
[sign4]: ../traffic-signs-data/sign4.jpg "Traffic Sign 4"
[sign5]: ../traffic-signs-data/sign5.jpg "Traffic Sign 5"

### Data Set Summary & Exploration
A summary of the traffic signs data set is shown in

![][dataset-summary]

_Dataset summary_

A visual inspection of the unique signs of the dataset is shown in

![dataset-visual-histogram]

_Visual histogram of unique signs in the dataset_


### Design and Test a Model Architecture

#### Normalization
The only pre-processing step applied was that of normalization. We normalized the dataset images by dividing each pixel by 255 and subtracting 0.5 from the result.

#### Model Architecture
The final model architecture is shown in the following table. The model was implemented in Keras with a TensorFlow (TF) backend.


| Layer         		|     Description in Keras API	        					|
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3)))    |
| Max Pooling					| model.add(MaxPooling2D((2, 2)))												|
| Dropout					|	model.add(Dropout(0.2))											|
| Convolution 1x1     	| model.add(Convolution2D(100, 1, 1)))    |
| RELU					|	model.add(Activation('relu'))											|
| Flatten	      	| model.add(Flatten())				|
| Fully Connected    |     		model.add(Dense(1024))							|
| RELU					|	model.add(Activation('relu'))
| Fully Connected    |     		model.add(Dense(256))							|
| RELU					|	model.add(Activation('relu'))											|
| Fully connected		| model.add(Dense(43))	        									|
| Softmax				| model.add(Activation('softmax'))        									|

Initially we chose a 5-stage architecture that involved a single convolution layer followed by regularization stages and two dense layers. RELU activations were used throughout. The validation performance of this network was 91.1%. We decided to try a deep convolutional 1x1 layer and increase the number of the dense layers to three as well as increase the span of first layer from 512 to 1024. Because we have seen small difference between the validation and test set performance we also reduce the dropout from 50% oto 20%.


#### Model Training
The approach adopted involved various steps that involved optimizing the parameters of the model (hyperparameters) given the architecture outlined in the previous section.

To help the search process we used sklearn's ```GridSearchCV``` feature. A dictionary of hyperparameters (```param_grid``` argument) must be provided representing the map of the model parameter name and an array of values to try. By default, accuracy is the score that is optimized. Cross validation is used to evaluate each individual model - the default of 3-fold cross validation is used.

In all experiments to maximize as much as possible repeatability of the results we fixed the random seed of numpy.

My final model results were:
* validation set accuracy of 94.830%
* test set accuracy of 92.36%

We did not try to exceed the tatget 93% accuracy as this would means significantly long (and expensive) runs on AWS.

#### Cross-validation Strategy
We should point out that there are two options for CV in problems where both architecture and hyperparameters need to determined. The first option involves at least one *nested* cross-validation loop as shown in

![][nested-cv]

_Nested CV with k\_{inner} and k\_{outer} folds. The inner CV loop determines via grid search the optimal hyper parameters and ensures that the selected permutation generalizes well on the inner loop validation data. The outer loop evaluates the model based on the generalization results on the outer validation data._

![][simple-cv]

_Simpler k-fold CV where the hyper parameters are determined in one step together with the other model parametes (e.g. weights)._

We elected to do a simple k-fold CV setting hyperparameters as well as testing various architectures in ad-hoc experiments.

#### Optimization of batch size and Number of Epochs
The optimal batch size was found to be 128 and we used 50 epochs to report the results.


### Test a Model on New Images and on the Test set
Here are five German traffic signs that we found on the web:

![alt text][sign1] ![alt text][sign2] ![alt text][sign3]
![alt text][sign4] ![alt text][sign5]


The results on the test set are as follows:

Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


