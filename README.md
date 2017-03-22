# Vehicle Detection

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Project requirements:

Project uses Python 3.5+ with libraries:
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

Example images of cars:

![img](output_images/veh1.png)
![img](output_images/veh2.png)

Example images of non-cars:

![img](output_images/nei1.png)
![img](output_images/nei2.png)
![img](output_images/nei3.png)

#### Features extraction
In order to train [LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) I had to extract features from images in data set, as images are not allowed directly as input for classifier.
However Linear SVC accepts long vectors. To get vectors from images I managed to extract:

* Histogram features of YCrCb Colorspace

* HOG features

* Spatial features


##### Colorspace

Durring testing best result vere given for YCrCb and HSL. In both of the color spaces value grows with luminance. And it make sense as cars are really like to be shiny
This color space was used to extract histogram of colors and as values to calculate HOG

##### HOG features
I managed to use Histogram of Oriented Gradients (HOG) with fallowing parameters:

    colorspace = 'YCrCb' 
    orient = 12
    pix_per_cell = 8
    cell_per_block = 2

Values on first where set by intuition - what looks good (easy to see diference) on test images should also be good for SVC.
Later on during train process orient was shifted from 8 to 12 to maximalise train effect (from 99.02% to 99.58%)

Results:  
orignal/hog

![img](output_images/veh1.png)
![img](output_images/veh1hog.png)

![img](output_images/veh2.png)
![img](output_images/veh2hog.png)

![img](output_images/nei1.png)
![img](output_images/nei1hog.png)

![img](output_images/nei2.png)
![img](output_images/nei2hog.png)

![img](output_images/nei3.png)
![img](output_images/nei3hog.png)


It's quite easy to see, that cars have a lot of squres, while non cars have different shapes





### Training features
All 18k images were used to train SVC 




### Sliding window


### HOG optimalisation



As hog is very processing power consumming it is really efficient aproach to calculet HOG once for the whole image, and later only reuse these values:
![img](output_images/hog-sub.jpg)


    # Compute individual channel HOG features for the entire image
    hog1 = train.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = train.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = train.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    

    (...) And later for sub images:
    
    hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
    hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
    hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

    test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                

### Heatmap to reduce false detections


### Conclusion