# **Traffic Sign Classifier**

Pantelis Monogioudis

NOKIA

## Summary
For German traffic signs classification, the Keras API was used to define a deep neural network and train its hyperparameters. For optimizing the hyperparameters of the model, the scikit-learn python library search API was used as described in the model training section. We have investigated various architectures starting from the LeNet network and adding overfitting mitigation and complexity reduction layers. The performance of the test dataset as well as the performance of random 5 images collected from the internet shows that the selected architecture as well as the hyperparameter search (although not exhaustive) performs well and can be easily replicated by cloning the repo: the [project code](https://github.com/pantelis/traffic-sign-classifier) and following the README.md instructions.  The later repo must contain the unzipped [downloaded data](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip) for the project code to work.

The goals / steps of this project are the following:

* Load, Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

[//]: # (Image References)

[dataset-visual-histogram]: ./traffic-signs-visual-histogram.png "Visual Histogram"
[dataset-summary]: ./traffic-signs-dataset-statistics.png "Dataset Summary"
[dataset-visual-histogram]: ./traffic-signs-visual-histogram.png "Visual Histogram"
[nested-cv]: ./nested-cv.jpg "Nested CV"
[simple-cv]: ./simple-cv.jpg "Simple CV"
[internet-german-signs]: ./internet-german-signs.png "Internet German Traffic Signs"
[softmax-output]: ./softmax-output.png "Prediction Probabilities"

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

Initially we chose a multi-stage architecture implemented entirely in TF as contained in the file ```LeNet-Traffic-Sign-Classifier.py``` that involved two convolution layers each followed by regularization stages and two dense layers. RELU activations were used throughout. The validation performance of this network was 91.1%. We decided to try a deep convolutional 1x1 layer and increase the number of the dense layers to three as well as increase the span of first layer from 512 to 1024. Because we have seen small difference between the validation and test set performance we also reduce the dropout from 50% oto 20%.


#### Model Training
The approach adopted involved various steps that involved optimizing the parameters of the model (hyperparameters) given the architecture outlined in the previous section.

In all experiments to maximize as much as possible repeatability of the results we fixed the random seed of numpy. The best batch size was emperically found to be 128 and 50 epochs were used to report the results.

The code can demonstrate the following results:

* validation set accuracy of 94.830%

We did not try to exceed the tatget 93% accuracy as this would means significantly long (and expensive) runs on AWS. All runs were completed after hours of execution in a quad-core CPU-only laptop.

The implementation saved the model into a JSON file and also saves a checkpoint every time the accuracy improved throughput the run. This means that a session can be restored and the trained model used to recognise other images as described next.


### Test a Model on New Images
We have run the trained model on the test set portion of the same dataset and observed an accuracy of 92.53%.

![][internet-german-signs]

We also did a google search on german traffic signs and as shown in the figure they need preprocessing so that they can be fed in the trained network pipeline. The single additional preprocessing stage applied was scaling that results in signs of equal size. This is shown in the second row of images in the figure

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60 kmph limit      		| 60 kmph limit   									|
| 50 kmph limit		|  50 kmph limit									|
| No passing					| Ahead only											|
| Stop sign      		| Stop sign				 				|
| Priority road			| Priority road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Obviously the small test set does not allow extrapolation to the much larger test result that achieved 92.53% test accuracy

The probabilities for the first three images are shown in the figure below:

![][softmax-output]

