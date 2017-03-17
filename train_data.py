import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog



#Some global parameter
hog_pix_per_cell = 8
hog_cell_per_block = 2
hog_orient = 12


def get_train_images():
    vehicles_paths = glob.glob('train_data/vehicles/**/*.png')
    non_vehicles_paths = glob.glob('train_data/vehicles/**/*.png')
    return vehicles_paths, non_vehicles_paths


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    # Use cv2.resize().ravel() to create the feature vector
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    return cv2.resize(feature_image, size).ravel()


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    c1hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    c2hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    c3hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = c1hist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((c1hist[0], c2hist[0], c3hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return c1hist, c2hist, c3hist, bin_centers, hist_features


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vec)
        return features


def get_histogram_features(img):
    pass



def get_img_features(colr_img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    orient = 8
    pix_per_cell = 8
    cell_per_block = 2
    # Call our function with vis=True to see an image output
    hog_features = get_hog_features(gray, orient,
                                           pix_per_cell, cell_per_block,
                                           vis=False, feature_vec=False)

    print(hog_features)
    # plt.imshow(hog_image,cmap="gray")
    # plt.show()
    # mpimg.imsave("report/veh2hog.png",hog_image )




img = mpimg.imread("report/veh2.png")


get_img_features(img)


feature_vec = bin_spatial(img, color_space='RGB', size=(32, 32))

#compose histogram features with with hog features
