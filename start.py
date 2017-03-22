import cv2
import train
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import settings
import glob


TRAIN = settings.TRAIN

svc_filename = 'svc_model.p'
if TRAIN:
    svc_pickle = train.train_data()
    train.save_train_model(svc_pickle,svc_filename)
else:
    svc_pickle = train.load_svc_model(svc_filename)


print(svc_pickle)
svc = svc_pickle["svc"]
X_scaler = svc_pickle["scaler"]
orient = svc_pickle["orient"]
pix_per_cell = svc_pickle["pix_per_cell"]
cell_per_block = svc_pickle["cell_per_block"]
spatial_size = svc_pickle["spatial_size"]
hist_bins = svc_pickle["hist_bins"]


def slide_window(img, classifier, y_start, y_stop, x_start, x_stop, overlay,show_rectangles = False):

    #Her i will store the images that contain all the result boxes
    boxes = []

    y_size = y_stop - y_start
    x_size = x_stop - x_start
    window_size = (y_size,y_size)
    step_size = y_size - int(y_size * overlay)
    steps = int(x_size / step_size)
    print(x_size,y_size,step_size,steps)
    for step_id in range(steps):
        x_left = (x_start + step_size*step_id)
        x_right = (x_start + step_size*step_id + y_size)
        y_top = y_start
        y_bot = y_stop
        x_max = x_stop


        if( x_right > x_max):
            x_right = x_max
            x_left = x_max - y_size

        bbox = (x_left, y_bot), (x_right, y_top)
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        thick = 7
        if show_rectangles:
            cv2.rectangle(img, bbox[0], bbox[1], color, thick)

        subimg = img[y_top:y_bot,x_left:x_right,:]
        feature_img = cv2.resize(subimg, (64, 64))
        concentrated_features = train.img_features(feature_img)
        concentrated_features = X_scaler.transform(concentrated_features)
        sub_score = (classifier.predict(concentrated_features))
        if (sub_score[0] > 0):
            print("Found")
            boxes.append(bbox)
    return boxes











def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    boxes = []

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = train.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = train.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = train.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = train.bin_spatial(subimg, size=spatial_size)
            hist_features = train.color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            #features.append(np.concatenate((spatial_features, hist_features, hog_features)))
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                #cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                boxes.append([(xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart)])

    return draw_img, boxes


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=3):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img


def draw_boxes_list(img, bboxes_list, color=(0, 0, 255), thick=3):
    # Make a copy of the image
    draw_img = np.copy(img)
    for bboxes in bboxes_list:

        # Iterate through the bounding boxes
        for bbox in bboxes:
            #color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
    return draw_img


def history_to_single_list():
    bbox_list = []
    for history in heatmap_history:
        for box in history:
            bbox_list.append(box)
    print(bbox_list)
    return bbox_list


# cap = cv2.VideoCapture('project_video.mp4')
# cap = cv2.VideoCapture('test_video.mp4')



def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[1][1]:box[0][1], box[0][0]:box[1][0]] += 1
        # print("aloha",box[0][1],box[1][1], box[0][0],box[1][0] )


    # Return updated heatmap
    # plt.imshow(heatmap)
    # plt.show()
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap



def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

from scipy.ndimage.measurements import label




# plt.imshow(draw_img)
# plt.show()

heatmap_history = []
heatmap_history_length = 40

def append_heatmap_history(bboxes):
    heatmap_history.append(bboxes)
    if len(heatmap_history) > heatmap_history_length:
        del heatmap_history[0]



#on my ubuntu machine
frame_id = 0
images = sorted(glob.glob('in_images/video_*.png'))
test_range_min = 1030
test_range_max = 2700
for x in images:
    frame_id += 1
    if frame_id < test_range_min or frame_id > test_range_max:
        continue

    in_img = cv2.imread(x)

    boxes1 = slide_window(in_img, svc, 430, 660, 40, 1250, 0.5)
    boxes2 = slide_window(in_img, svc, 388, 578, 40, 1250, 0.5)
    boxes3 = slide_window(in_img, svc, 408, 524, 40, 1250, 0.5)
    boxes4 = slide_window(in_img, svc, 408, 490, 40, 1250, 0.5)
    boxes5 = slide_window(in_img, svc, 408, 460, 40, 1250, 0.5)
    boxes6 = slide_window(in_img, svc, 412, 468, 40, 1250, 0.5)

    BOXES = []
    BOXES.append(boxes1)
    BOXES.append(boxes2)
    BOXES.append(boxes3)
    BOXES.append(boxes4)
    BOXES.append(boxes5)
    BOXES.append(boxes6)

    boxes_img = draw_boxes_list(in_img,BOXES)
    cv2.imshow("boxews", boxes_img)

    append_heatmap_history(boxes1)
    append_heatmap_history(boxes2)
    append_heatmap_history(boxes3)
    append_heatmap_history(boxes4)
    append_heatmap_history(boxes5)
    append_heatmap_history(boxes6)

    heat_boxes = history_to_single_list()

    heat = np.zeros_like(in_img[:, :, 0]).astype(np.float)
    heatmap = add_heat(heat,heat_boxes)
    cv2.imshow("heatmap", heatmap)
    plt.imshow(heatmap)
    plt.show()
    heatmap = apply_threshold(heatmap, 2)

    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)

    draw_img = draw_labeled_bboxes(np.copy(in_img), labels)




    out = draw_boxes_list(in_img,BOXES,(0,0,255),3)
    cv2.imshow("frame",draw_img)








    filename = ("video/frame_{:04d}.png".format(frame_id))
    cv2.imwrite(filename,out)
    filename = ("heatmap/frame_{:04d}.png".format(frame_id))
    cv2.imwrite(filename,draw_img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("report/sample_pre.png", in_img)
        cv2.imwrite("report/sample_out.png", out)
        break



# On my windows machine
# frame_id = 0
# while(cap.isOpened()):
#
#
#     ret, out = cap.read()
#     in_img = out
#     if ret==True:
#
#
#
#         boxes1 = slide_window(out, svc, 430, 660, 40, 1250, 0.5)
#         boxes2 = slide_window(out, svc, 388, 578, 40, 1250, 0.5)
#         boxes3 = slide_window(out, svc, 408, 524, 40, 1250, 0.5)
#         boxes4 = slide_window(out, svc, 408, 490, 40, 1250, 0.5)
#         boxes5 = slide_window(out, svc, 408, 460, 40, 1250, 0.5)
#
#         BOXES = []
#         BOXES.append(boxes1)
#         BOXES.append(boxes2)
#         BOXES.append(boxes3)
#         BOXES.append(boxes4)
#
#         out = draw_boxes_list(in_img,BOXES,(0,0,255),3)
#         cv2.imshow("frame",out)
#
#
#
#
#
#
#
#         filename = ("video/frame_{:04d}.png".format(frame_id))
#         cv2.imwrite(filename,out)
#         frame_id += 1
#
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cv2.imwrite("report/sample_pre.png", in_img)
#             cv2.imwrite("report/sample_out.png", out)
#             break
#     else:
#         break
#
# cap.release()
#cv2.destroyAllWindows()