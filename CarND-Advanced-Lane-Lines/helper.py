import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def calibrate(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def warp(img):
    img_size = (img.shape[1], img.shape[0])
    #Coordinates outlining the lane
    tl_src = [562, 471]
    tr_src = [720, 471]
    br_src = [1088, 720]
    bl_src = [206, 720]
    
    # determine the height and width of the transformed image
    widthA = np.sqrt(((br_src[0] - bl_src[0]) ** 2) + ((br_src[1] - bl_src[1]) ** 2))
    widthB = np.sqrt(((tr_src[0] - tl_src[0]) ** 2) + ((tr_src[1] - tl_src[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr_src[0] - br_src[0]) ** 2) + ((tr_src[1] - br_src[1]) ** 2))
    heightB = np.sqrt(((tl_src[0] - bl_src[0]) ** 2) + ((tl_src[1] - bl_src[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    #Coordinates of the transformed image
    '''
    tl_dst = [0, 0]
    tr_dst = [maxWidth - 1, 0]
    br_dst = [maxWidth - 1, maxHeight - 1]
    bl_dst = [0, maxHeight - 1]
    
    tl_dst = [0, 0]
    tr_dst = [img.shape[1], 0]
    br_dst = [img.shape[1]-1, img.shape[0]-1]
    bl_dst = [0, img.shape[0]-1]
    '''
    tl_dst = [200,270]
    tr_dst = [900,270]
    br_dst = [900,720]
    bl_dst = [200,720]
    src = np.float32(
        [tl_src,
         tr_src,
         br_src,
         bl_src])
    
    dst = np.float32([tl_dst,
                      tr_dst, 
                      br_dst, 
                      bl_dst])
    
    # Compute the perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    
    # Compute the inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    #cv2.polylines(img,np.int32([src]),True,(255,0,0), 5)
    #cv2.polylines(img,np.int32([dst]),True,(0,0,255), 5)
    #f, ax = plt.subplots(1, 1, figsize=(20,10))
    #ax.imshow(img)
    return warped, Minv


def getSobelBinaryX(gray, sobel_kernel = -1, thresh_min = 0, thresh_max=255):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    abs_sobelx = np.absolute(sobelx) 
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    scaled_sobelx = cv2.equalizeHist(scaled_sobelx)
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= thresh_min) & (scaled_sobelx <= thresh_max)] = 1
    return sxbinary

def getSobelBinaryY(gray, sobel_kernel = -1, thresh_min = 0, thresh_max=255):
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    abs_sobely = np.absolute(sobely)
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
    scaled_sobely = cv2.equalizeHist(scaled_sobely)
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= thresh_min) & (scaled_sobely <= thresh_max)] = 1
    return sybinary

def getLaplace(gray, thresh_min = 0, thresh_max=255):
    laplace = cv2.Laplacian(gray, cv2.CV_64F)
    abs_laplace = np.absolute(laplace)
    scaled_laplace = np.uint8(255*abs_laplace/np.max(abs_laplace))
    scaled_laplace = cv2.equalizeHist(scaled_laplace)
    lapbinary = np.zeros_like(scaled_laplace)
    lapbinary[(scaled_laplace >= thresh_min) & (scaled_laplace <= thresh_max)] = 1
    return lapbinary

def getSobelX(gray, sobel_kernel=3):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel) 
    abs_sobelx = np.absolute(sobelx) 
    return abs_sobelx
    
def getSobelY(gray, sobel_kernel=3):
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    abs_sobely = np.absolute(sobely)
    return abs_sobely
    
def getSobelDirection(gray, sobel_kernel = 3, thresh_min = 0, thresh_max=np.pi/2):
    sobelx = getSobelX(gray, sobel_kernel=sobel_kernel)
    sobely = getSobelY(gray, sobel_kernel=sobel_kernel)
    direction = np.arctan2(sobely, sobelx)
    dirbinary = np.zeros_like(direction)
    dirbinary[(direction >= thresh_min) & (direction <= thresh_max)] = 1
    return dirbinary
    
def getSatBinary(s_channel, thresh_min = 150, thresh_max=255):
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh_min) & (s_channel <= thresh_max)] = 1
    return s_binary

def combineBinary(combined, *binary):
    for b in binary:
        combined[(combined == 1) | (b == 1)] = 1
    return combined

def getThres(gray, thresh_min = 0, thresh_max=255):
    binary = np.zeros_like(gray)
    binary[(gray >= thresh_min) & (gray <= thresh_max)] = 1
    return binary

def windSlide(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 90
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_top = binary_warped.shape[0] - (window+1)*window_height
        win_y_bot = binary_warped.shape[0] - window*window_height
        win_xleft_bot = leftx_current - margin
        win_xleft_top = leftx_current + margin
        win_xright_bot = rightx_current - margin
        win_xright_top = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_bot,win_y_bot),(win_xleft_top,win_y_top),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_bot,win_y_bot),(win_xright_top,win_y_top),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_top) & (nonzeroy < win_y_bot) & 
        (nonzerox >= win_xleft_bot) &  (nonzerox < win_xleft_top)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_top) & (nonzeroy < win_y_bot) & 
        (nonzerox >= win_xright_bot) &  (nonzerox < win_xright_top)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    #fig, ax = plt.subplots()
    #ax.imshow(out_img)
    #ax.plot(left_fitx, ploty, color='yellow')
    #ax.plot(right_fitx, ploty, color='yellow')
    #return left_fitx, right_fitx, ploty
    return left_fit, right_fit
    
def usePrevSlide(binary_warped, left_fit, right_fit,lanes):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    lane_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    lane_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    lane_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    all_line_pts = np.hstack((left_line_window1, right_line_window2))
    cv2.fillPoly(window_img, np.int_([all_line_pts]), (0,255, 0))
    result = cv2.addWeighted(lane_img, 1, window_img, 1, 0)
    
    
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    lanes.left_curverad = left_curverad
    lanes.right_curverad = right_curverad
    #fig, ax = plt.subplots()
    #ax.imshow(out_img)
    #ax.plot(left_fitx, ploty, color='yellow')
    #ax.plot(right_fitx, ploty, color='yellow')
    return result , left_fit, right_fit