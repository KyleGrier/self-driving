import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
import time
from collections import deque


def getData(): 
    # Read in cars and notcars
    # images = glob.glob('prac_img\\smallset/*/*/*.jpeg')
    images = glob.glob('dataset/**/*.png', recursive=True)
    cars = []
    notcars = []
    for image in images:
        if "non-vehicles" in image:
            notcars.append(image)
        else:
            cars.append(image)
    return cars, notcars

# Define a function that takes an image, a list of bounding boxes, 
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    for p1, p2 in bboxes:
        cv2.rectangle(draw_img, p1, p2, color, thick)
    # return the image copy with boxes drawn
    return draw_img

# Define a function that takes an image, a list of bounding boxes, 
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output
def draw_boxes_visual(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    for p1, p2 in bboxes:
        clone = np.copy(img)
        cv2.rectangle(clone, p1, p2, color, thick)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.05)
        cv2.rectangle(draw_img, p1, p2, color, thick)
    # return the image copy with boxes drawn
    return draw_img

# Convert from RGB to the specified color space
def color_config(img, cspace='RGB'):
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    return feature_image 

def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    # Use cv2.resize().ravel() to create the feature vector
    img = color_config(img, cspace=color_space)  
    img = cv2.resize(img, size)
    features = img.ravel()
    # Return the feature vector
    return features

#Gets HOG features and can return a visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False,
                     feature_vec=True):
                         
    """
    Function accepts params and returns HOG features (optionally flattened) and an optional matrix for 
    visualization. Features will always be the first return (flattened if feature_vector= True).
    A visualization matrix will be the second return if visualize = True.
    """
    return hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm= 'L2-Hys', transform_sqrt=False, 
                                  visualise= vis, feature_vector= feature_vec)

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), hog_channel="ALL",
                        orient=9, pix_per_cell=8, cell_per_block=2,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        feat_list = []
        # Read in each one by one
        img = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        img = color_config(img, cspace=cspace)     
        # Apply bin_spatial() to get spatial color features
        img = cv2.resize(img, (64, 64))
        rgb_img = color_config(img, cspace='RGB')
        #3) Compute spatial features if flag is set
        if spatial_feat == True:
            spat_feat = cv2.resize(rgb_img, spatial_size)
            spat_feat = spat_feat.ravel()
            feat_list.append(spat_feat)
        #5) Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = color_hist(rgb_img, nbins=hist_bins)
            #6) Append features to list
            feat_list.append(hist_features)
        #7) Compute HOG features if flag is set
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(img.shape[2]):
                    hog_features.append(get_hog_features(img[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(img[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block)
            feat_list.append(hog_features)
        # Append the new feature vector to the features list
        if len(feat_list) == 1:
            features.append(feat_list[0])
        else:
            features.append(np.concatenate(feat_list))
    # Return list of feature vectors
    return features

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel="ALL",
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    feat_img = color_config(img, cspace=cspace)
    feat_img = cv2.resize(feat_img, (64, 64))  
    rgb_img = color_config(feat_img, cspace='RGB')   
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        #4) Append features to list
        spat_feat = cv2.resize(rgb_img, spatial_size)
        spat_feat = spat_feat.ravel()
        img_features.append(spat_feat)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(rgb_img, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feat_img.shape[2]):
                hog_features.append(get_hog_features(feat_img[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feat_img[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block)
        img_features.append(hog_features)
    if len(img_features) == 1:
        return np.array(img_features[0])
    else:
        #9) Return concatenated array of features
        return np.concatenate(img_features)

    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, cspace='LUV', 
                    spatial_size=(64, 64), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel="ALL", spatial_feat=False, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #print(len(windows))
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]] 
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, cspace=cspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    #print(len(on_windows))
    return on_windows

def save_on_windows(img, windows, id, num):
    for i, window in enumerate(windows):
        filename = "hard_negative/tosave{}.png".format(str(id) + "_" + str(num) +"_"+ str(i))
        #3) Extract the test window from original image
        save_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        save_img = (save_img*255.0).astype("uint8")
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, save_img)

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    bboxes = []
    areas = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
        area = Frame.get_area(bbox)
        areas.append(area)
    #Remove false positive
    new_areas = [] 
    new_bbox = []
    for area, box in zip(areas,bboxes):
        if area > 2000:
            new_bbox.append(box)
            new_areas.append(area)
    areas = new_areas
    bboxes = new_bbox
    if(len(bboxes) > 2):
        small = None
        small_area = None    
        for i, area in enumerate(areas):
            if small == None:
                small = i
                small_area = area
            elif(area < small_area):
                small_area = area
                small = i 
        del areas[small]
        del bboxes[small]      
    for area, box in zip(areas,bboxes):
        cv2.rectangle(img, box[0], box[1], (0,255,0), 6)
    # Iterate through all detected cars
    # for car_number in range(1, labels[1]+1):
    #     # Find pixels with each car_number label value
    #     nonzero = (labels[0] == car_number).nonzero()
    #     # Identify x and y values of those pixels
    #     nonzeroy = np.array(nonzero[0])
    #     nonzerox = np.array(nonzero[1])
    #     # Define a bounding box based on min/max x and y
    #     bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
    #     print(Frame.get_area(bbox))
    #     # Draw the box on the image
    #     cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)
    # Return the image
    return img

# class Containers():
#     def __init__(self):
#         # was the line detected in the last iteration?
#         frames = []
#     def add_frame(box):
#         new_frame = Frame()
#         new_frame.add_box(box)
#         frames.add(new_frame)

#     def add_boxes(boxes):
#         prosp_list = 
#         frame_list = list(self.frames)
#         matches = []
#         for frame in frame_list:
#             best_fit  = np.nan
#             best_dist = 0
#             for prosp in prosp_list:
#                 if (best_fit is np.nan):
#                     best_fit = prosp
#                     best_dist = abs(frame.xmin - prosp[0][0])
#                 else:
#                     new_dist = abs(frame.xmin - prosp[0][0])
#                     if new_dist < best_dist:
#                         best_dist = new_dist
#                         best_fit = prosp



class Frame():
    def __init__(self, outline):
        self.frames = deque()
        # Average area of boxes for the previous 5 labels.
        self.avg = 0
        self.max_avg = 0
        self.xmin = 0
        #the left most x value of the outlines
        self.left = 0
    # def add_box(box):
    #     if box !
    #     area = get_area(box)
    #     self.frames.appendleft(area)
    #     if (len(self.frames) > self.max_avg):
    #         self.frames.pop() 
    #     self.avg = np.nanmean(self.frames)
    #     if 
    #     self.xmin = box[0][0]
    def get_area(box):
        l = box[1][0] - box[0][0]
        h = box[1][1] - box[0][1]
        return l * h
    def get_xmin():
        return self.xmin
    def can_remove():
        return 

#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################


	
if __name__ == "__main__":
	# performs under different binning scenarios
	spatial = 32
	histbin = 32
	# Define a labels vector based on features lists
	y = np.hstack((np.ones(len(car_features)), 
	              np.zeros(len(notcar_features))))
	# Create an array stack of feature vectors
	X = np.vstack((car_features, notcar_features)).astype(np.float64)
	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(
    	X, y, test_size=0.2, random_state=rand_state)
	# Fit a per-column scaler only on the training data
	X_scaler = StandardScaler().fit(X_train)
	# Apply the scaler to both X_train and X_test
	scaled_X_train = X_scaler.transform(X_train)
	scaled_X_test = X_scaler.transform(X_test)






#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################



# Define a function to search for template matches
# and return a list of bounding boxes
def find_matches(img, template_list):
    # Define an empty list to take bbox coords
    bbox_list = []
    # Define matching method
    # Other options include: cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCORR',
    #         'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
    method = cv2.TM_CCOEFF_NORMED
    # Iterate through template list
    for temp in template_list:
        # Read in templates one by one
        tmp = mpimg.imread("prac_img/cutouts/" + temp)
        # Use cv2.matchTemplate() to search the image
        result = cv2.matchTemplate(img, tmp, method)
        # Use cv2.minMaxLoc() to extract the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # Determine a bounding box for the match
        w, h = (tmp.shape[1], tmp.shape[0])
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # Append bbox position to list
        bbox_list.append((top_left, bottom_right))
        # Return the list of bounding boxes
        
    return bbox_list

def doTemplate():
	bboxes = find_matches(image, templist)
	result = draw_boxes(image, bboxes)
	plt.imshow(result)
	plt.savefig("prac_img/template_matching")

def plot3d(pixels, colors_rgb,
        axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255))):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation


def do_color_hist():
	image = mpimg.imread('prac_img/cutouts/cutout1.jpg')
	rh, gh, bh, bincen, feature_vec = color_hist(image, nbins=32, bins_range=(0, 256))

	# Plot a figure with all three bar charts
	if rh is not None:
	    fig = plt.figure(figsize=(12,3))
	    plt.subplot(131)
	    plt.bar(bincen, rh[0])
	    plt.xlim(0, 256)
	    plt.title('R Histogram')
	    plt.subplot(132)
	    plt.bar(bincen, gh[0])
	    plt.xlim(0, 256)
	    plt.title('G Histogram')
	    plt.subplot(133)
	    plt.bar(bincen, bh[0])
	    plt.xlim(0, 256)
	    plt.title('B Histogram')
	    fig.tight_layout()
	    plt.savefig("prac_img/color_histo")
	else:
	    print('Your function is returning None for at least one variable...')

def doFeatExtract():
    images = glob.glob('prac_img\\smallset/*/*/*.jpeg')
    cars = []
    notcars = []
    for image in images:
        if 'image' in image or 'extra' in image:
            notcars.append(image)
        else:
            cars.append(image)

    car_features = extract_features(cars, cspace='RGB', spatial_size=(32, 32),
                            hist_bins=32, hist_range=(0, 256))
    notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(32, 32),
                            hist_bins=32, hist_range=(0, 256))

    if len(car_features) > 0:
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        car_ind = np.random.randint(0, len(cars))
        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(cars[car_ind]))
        plt.title('Original Image')
        plt.subplot(132)
        plt.plot(X[car_ind])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(scaled_X[car_ind])
        plt.title('Normalized Features')
        fig.tight_layout()
    else: 
        print('Your function only returns empty feature vectors...')

# Define a function to return some characteristics of the dataset 
# Pass in the path to the car and notcar images for the dataset
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict


    # if len(frames) == 0:
    #     for box in bboxes: 
    #         new_frame = frame()
    #         new_frame.add_box(box)
    # elif len(frames) == 1:
    #     if len(bbox) == 0:
    #         frames[0].add_box(np.nan)
    #     elif len(bbox == 1):
    #         frames[0].add_box(bbox[0])
    #     else:
    #         comp = frames[0].get_minx()
    #         box1_dist = abs(bbox[0][0][0] - comp)
    #         box2_dist = abs(bbox[1][0][0] - comp)
    #         if box1_dist > box2_dist:
    #             frames[0].add_box[bbox[1]]
    #             new_frame = Frame()
    #             new_frame.add_box[bbox[0]]
    #         else:
    #             frames[0].add_box[bbox[0]]
    #             new_frame = Frame()
    #             new_frame.add_box[bbox[1]]
    # else:
    #     if len(bbox) == 0:
    #         for fr in frames:
    #             fr.add_box(np.nan)
    #     elif len(bbox == 1):
    #         frames[0].add_box(bbox[0])
    #     else:
    #         comp = frames[0].get_minx()
    #         box1_dist = abs(bbox[0][0][0] - comp)
    #         box2_dist = abs(bbox[1][0][0] - comp)
    #         if box1_dist > box2_dist:
    #             frames[0].add_box[bbox[1]]
    #             new_frame = Frame()
    #             new_frame.add_box[bbox[0]]
    #         else:
    #             frames[0].add_box[bbox[0]]
    #             new_frame = Frame()
    #             new_frame.add_box[bbox[1]]