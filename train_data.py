import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split



#Some macro parameter
hog_pix_per_cell = 8
hog_cell_per_block = 2
hog_orient = 12
hog_target_size = (64,64)


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
        elif color_space == 'gray':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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





def get_img_features(colr_img):

    features = []

    copy = np.copy(colr_img)
    copy = cv2.resize(copy,hog_target_size)

    #gray = cv2.cvtColor(copy, cv2.COLOR_RGB2HLS)[:, :, 0]
    gray = cv2.cvtColor(copy, cv2.COLOR_RGB2GRAY)
    #ensure every image is same size

    orient = hog_orient
    pix_per_cell = hog_pix_per_cell
    cell_per_block = hog_cell_per_block
    # Call our function with vis=True to see an image output
    hog_features = get_hog_features(gray, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)


    hls_features = bin_spatial(colr_img,"HLS")
    #rgb_features = bin_spatial(colr_img)
    # print(hog_features)
    # print(len(rgb_features))
    # print(hog_features)

    return np.concatenate((hls_features,hog_features))




    # hog_features,hog_image = get_hog_features(gray, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
    # print(hog_features)
    # plt.imshow(hog_image,cmap="gray")
    # plt.show()
    # mpimg.imsave("report/x.png",hog_image )




def train_classifier():

    ts = time.time()
    car_features = []
    notcar_features = []

    vehicles_paths, non_vehicles_paths = get_train_images()
    print(round(time.time() - ts, 2), 'Images Loaded')

    for f in vehicles_paths:
        img =  mpimg.imread(f)
        car_features.append(get_img_features(img))

    print(round(time.time() - ts, 2), 'vehicles features extracted')

    for f in non_vehicles_paths:
        img =  mpimg.imread(f)
        notcar_features.append(get_img_features(img))

    print(round(time.time() - ts, 2), 'NON vehicles features extracted')

    y = np.hstack((np.ones(len(car_features)),
                   np.zeros(len(notcar_features))))

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    print(round(time.time() - ts, 2), 'Data Scaled')
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print(round(time.time() - ts, 2), 'Data splited')

    #print('Using:', orient, 'orientations', pix_per_cell,          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(time.time() - ts, 2), 'Data trainned - SVC')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts    : ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')


train_classifier()

#compose histogram features with with hog features
