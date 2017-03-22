# Vehicle Detection

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Project requirements:

Project is standalone unless you have Python 3.5+ with libraries:
* openCV 3.0
* skit-learn


### About this document:

This document should be useful for anyone who would like to known more about project, or anyoune who would like to understand pipline for robust vehicle detection from camera on mounted on a car.


The Project
---

#### Result 
You can see result [video](result.mp4) in this repo. Or watch video on [youtube](https://youtu.be/1STnCs36MfU)

Brief desciption how to get this effect:
1) Train LinearSVC for two groups: Car and Non-car
2) Slide window on wide input image. If curent window was classified as a Car mark it on image


#### Data set
Data set was included in this [repo](train_data) .
This data set counts over 18000 pictures (64x64 RGB) of 2 groups: vehicles and everything else what can be found on the road but it's not a vehicle 
These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

#### Features extraction
In order to train [LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) I had to extract features from images in data set, as images are not allowed directly as input for classifier.
However Linear SVC accepts long vectors. To get vectors from images I managed to extract:

* Histogram features of YCrCb Colorspace
* HOG features

##### Spatial features
