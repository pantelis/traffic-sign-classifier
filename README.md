## Project: Build a Traffic Sign Recognition Program

Overview
---
We train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

We have implemented using two python files - one trains the model as well as produces the accuracy performance metric over the test dataset and the other loads the saved model from disk and produces the softmax output probabilities and predictions over 5 images downloaded from the internet.

We have *not* used ipython notebooks and instructions on how to replicate the results are outlined below.

1. Clone the github repo https://github.com/pantelis/traffic-signs-data
2. [Download the data](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip) and extract the contents in the cloned repo directory.
3. Ensure that all the dependencies have been installed and you are in the environment outlined in the Dependencies section below.
4. Execute the ```keras-traffic-sign.py``` to train the model. At the end of the training the model json file ```keras-traffic-sign-model.json``` will be saved to disk. The file ```keras-traffic-sign-weights.h5``` containing the h5 formatted final weights will also be saved to disk.
5. Execute the ```internet-traffic-sign-prediction.py``` to produce the soft-max probabilities and the predictions against the 5 downloaded signs ```sign*.jpg```.

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.