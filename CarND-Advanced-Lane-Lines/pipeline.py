import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from helper import warp, getSobelBinaryX, getSobelBinaryY, getSobelX, \
                    getSobelY, getSobelDirection, getSatBinary, combineBinary, windSlide, usePrevSlide, calibrate



font                   = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText    = (10,100)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
def pipeline(img, objpoints, imgpoints,lanes):
    #Preprocess to create warped img
    img = calibrate(img, objpoints, imgpoints)
    img_size = (img.shape[1], img.shape[0])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    combined = s_channel* 0.6 + gray * 0.4
    combined = np.array( combined).astype ('uint8')
    blurf = np.zeros((1, 5))
    blurf.fill (1)
    combined = cv2.filter2D(combined, cv2.CV_32F, blurf)
    f = np.zeros((1, 30))
    f.fill (1)
    l = cv2.morphologyEx(combined, cv2.MORPH_OPEN, f)
    combined = combined - l
    combined = getSobelBinaryX(combined, sobel_kernel = 15, thresh_min = 190, thresh_max=255)
    binary_warped, minv = warp(combined)
    # Do window search or optimized window search using previous lanes
    
    # Unwarp the shaded lanes and apply to the calibrated image
    
    # Update the lane classes
    # Draw the lane onto the warped blank image
    if lanes.detected == False:
        lanes.left_fit, lanes.right_fit = windSlide(binary_warped)
        lanes.detected = True
    result, lanes.left_fit, lanes.right_fit = usePrevSlide(binary_warped, lanes.left_fit, lanes.right_fit, lanes)
    warpy = cv2.warpPerspective(result, minv, img_size, flags=cv2.INTER_LINEAR)
    final = cv2.addWeighted(img, 1, warpy, 1, 0)
    #cv2.putText(final,'Hello World!{}'.format(lanes.left_curverad),
    #            topLeftCornerOfText, 
    #            font, 
    #            fontScale,
    #            fontColor,
    #            lineType)
    return final