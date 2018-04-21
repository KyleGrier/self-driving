from helper import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import matplotlib.gridspec as gridspec
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import glob
import time
from scipy.ndimage.measurements import label

orient = 11
pix_per_cell = 16
cell_per_block = 2
cspace = 'YUV'
sig = "_ori"+ str(orient)+ "_pix" + str(pix_per_cell) + "_cell_per_block"
clf = None
scaler = None
test_image = "test_images/test6.jpg"
test_images = ["test_images_png/test0.png", "test_images_png/test1.png", "test_images_png/test2.png", "test_images_png/test3.png", "test_images_png/test4.png", "test_images_png/test5.png", "messigray.png"]

def saveFeatures(ori=orient, pix=pix_per_cell, cell=cell_per_block, cspace=cspace):

	start = time.time()

	cars, notcars = getData()
	start = time.time()
	car_features = extract_features(cars, cspace=cspace,
				orient = ori, pix_per_cell = pix, cell_per_block = cell)
	notcar_features = extract_features(notcars, cspace=cspace,
				orient = ori, pix_per_cell = pix, cell_per_block = cell)
	X = np.vstack((car_features, notcar_features)).astype(np.float64)
	y = np.hstack((np.ones(len(car_features)),
						np.zeros(len(notcar_features))))

	sig = "_ori"+ str(ori)+ "_pix" + str(pix) + "_cell_per_block" + str(cell) + "_cspace_" + cspace
	np.save("feats/X_data{}".format(sig), X)
	np.save("feats/y_data{}".format(sig), y)

	end = time.time() - start
	print("Saving the features took {} seconds".format(round(end, 2)))


	return

def loadData(ori=orient, pix=pix_per_cell, cell=cell_per_block, cspace=cspace):
	start = time.time()
	sig = "_ori"+ str(ori)+ "_pix" + str(pix) + "_cell_per_block" + str(cell) + "_cspace_" + cspace
	X = np.load("feats/X_data{}.npy".format(sig))
	y = np.load("feats/y_data{}.npy".format(sig))
	rand_state = np.random.randint(100)
	X_train, X_test, y_train, y_test = train_test_split(
		X, y,  test_size=0.1,  random_state= 1, stratify=y)

	end = time.time() - start

	print("Loading the features took {} seconds".format(round(end, 2)))

	return X_train, X_test, y_train, y_test

def loadDataFull(ori=orient, pix=pix_per_cell, cell=cell_per_block, cspace=cspace):
	start = time.time()
	sig = "_ori"+ str(ori)+ "_pix" + str(pix) + "_cell_per_block" + str(cell) + "_cspace_" + cspace
	X = np.load("feats/X_data{}.npy".format(sig))
	y = np.load("feats/y_data{}.npy".format(sig))

	end = time.time() - start

	print("Loading the features took {} seconds".format(round(end, 2)))

	return X, y

def trainModel(X_train, X_test, y_train, y_test):
	# Fit a per-column scaler only on the training data
	X_scaler = StandardScaler().fit(X_train)
	# Apply the scaler to X_train and X_test
	X_train = X_scaler.transform(X_train)
	X_test = X_scaler.transform(X_test)
	# Use a linear SVC 
	svc = LinearSVC(C=0.001)
	#svc = SVC(C=10) # best 100 only hog
	# Check the training time for the SVC
	t=time.time()
	svc.fit(X_train, y_train)
	pred = svc.predict(X_test)
	t2 = time.time()

	print(round(t2-t, 2), 'Seconds to train SVC...')
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
	tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
	print("Percent of False Positives = ", round(fp/len(X_test), 5))
	print("Percent of False Negatives = ", round(fn/len(X_test), 5))
	return svc, X_scaler

def trainModelFull(X, y):
	# Fit a per-column scaler only on the training data
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X_train and X_test
	X = X_scaler.transform(X)
	# Use a linear SVC 
	svc = LinearSVC(C=0.001)
	#svc = SVC(C=10) # best 100 only hog
	# Check the training time for the SVC
	t=time.time()
	svc.fit(X, y)
	t2 = time.time()

	return svc, X_scaler


if __name__ == "__main__":

	saveFeatures(ori=orient, pix=pix_per_cell, cell=cell_per_block, cspace=cspace)
	# X_train, X_test, y_train, y_test = loadData()
	# trainModel(X_train, X_test, y_train, y_test)
	X, y = loadDataFull()
	clf, scaler = trainModelFull(X, y)
	frames = []
	for i, img_name in enumerate(test_images):
		img = plt.imread(img_name)
		heat = np.zeros_like(img[:,:,0]).astype(np.float)

		windows = slide_window(img, x_start_stop=[650, img.shape[1]+20],
			y_start_stop=[380, 650], xy_overlap=(0.65, 0.65))
		windows2 = slide_window(img, x_start_stop=[650, img.shape[1]+40],
			y_start_stop=[380, 650], xy_overlap=(0.75, 0.75), xy_window=(80, 80))
		windows3 = slide_window(img, x_start_stop=[650, img.shape[1]+40],
			y_start_stop=[380, 650], xy_overlap=(0.75, 0.75), xy_window=(96, 96))

		t = time.time()
		on_windows = search_windows(img, windows, clf, scaler,
						orient=orient, pix_per_cell=pix_per_cell,
						cell_per_block=cell_per_block, cspace=cspace)
		t2 = time.time()
		# print(round(t2-t, 2), 'Seconds to perform search_windows')

		t = time.time()
		on_windows2 = search_windows(img, windows2, clf, scaler,
						orient=orient, pix_per_cell=pix_per_cell,
						cell_per_block=cell_per_block, cspace=cspace)
		t2 = time.time()
		# print(round(t2-t, 2), 'Seconds to perform search_windows')

		t = time.time()
		on_windows3 = search_windows(img, windows3, clf, scaler,
						orient=orient, pix_per_cell=pix_per_cell,
						cell_per_block=cell_per_block, cspace=cspace)
		t2 = time.time()
		# # print(round(t2-t, 2), 'Seconds to perform search_windows')

		final =  draw_boxes(img, windows)
		plt.imshow(final)
		plt.savefig("output_images/{}windows".format(i))

		final =  draw_boxes(img, windows2)
		plt.imshow(final)
		plt.savefig("output_images/{}windows2".format(i))

		final =  draw_boxes(img, windows3)
		plt.imshow(final)
		plt.savefig("output_images/{}windows3".format(i))

		final =  draw_boxes(img, on_windows)
		plt.imshow(final)
		plt.savefig("output_images/{}search_windows".format(i))

		final =  draw_boxes(img, on_windows2)
		plt.imshow(final)
		plt.savefig("output_images/{}search_windows2".format(i))

		final =  draw_boxes(img, on_windows3)
		plt.imshow(final)
		plt.savefig("output_images/{}search_windows3".format(i))

		save_on_windows(img, on_windows, 1, i)
		save_on_windows(img, on_windows2, 2, i)
		save_on_windows(img, on_windows3, 3, i)
        
		heatmap = add_heat(heat, on_windows)
		heatmap = add_heat(heatmap, on_windows2)
		heatmap = add_heat(heatmap, on_windows3)
		heatmap = apply_threshold(heatmap, 2)
		plt.imshow(heatmap, cmap='hot')
		plt.savefig("output_images/{}heatmap".format(i))
		labels = label(heatmap)
		draw_img = draw_labeled_bboxes(np.copy(img), labels)
		plt.imshow(draw_img)
		plt.savefig("output_images/{}box".format(i))
	print("img{} labels =".format(i) + str(labels[1]))
