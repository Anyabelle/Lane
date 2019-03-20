import os
from enum import IntEnum
import time
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML

img = cv2.imread('camera_cal/calibration1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_size = (img.shape[1], img.shape[0])

G_SHAPE = img.shape
G_W = G_SHAPE[1]
G_H = G_SHAPE[0]

#
# Find coners on chessboard images and prepare points for calibrateCamera
#
CB_X = 6
CB_Y = 9

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CB_X * CB_Y, 3), np.float32)
objp[:,:2] = np.mgrid[0 : CB_Y, 0 : CB_X].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    chess_img = cv2.imread(fname)
    chess_img_gray = cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(chess_img_gray, (CB_Y, CB_X), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
#
# Find camera calibration parameters given object points and image points
#
ret, G_MTX, G_DIST, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)


h, w = (720, 1280)
mapx, mapy = cv2.initUndistortRectifyMap(G_MTX, G_DIST, None, G_MTX, (w,h) , 5)
def undistort(img):
    
    #import time

    #start = time.time()
    #dst = cv2.undistort(img, G_MTX, G_DIST, None, G_MTX)
    
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    #end = time.time()
    #print(end - start)
    return dst

def channel_select(channel, thresh=(0, 255)):
    binary = np.zeros_like(channel)
    binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def sobel_thresh(channel, thresh=(30, 255)):
    sobel = cv2.Sobel(channel, cv2.CV_64F, 1, 0)
    #sobel = cv2.Laplacian(img,cv2.CV_64F)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def lightness(img):
    return np.uint8(img[:,:,0] * 0.299 + img[:,:,1] * 0.587 + img[:,:,2] * 0.1170)

import time
G_CLACHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

def make_binary_fast(img):
    img = np.copy(img)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    # get separate channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    gray = grayscale(img)
    S_channel = hls[:,:,2]

    S_filter = channel_select(S_channel, (220, 255))
    Sobel_filter = sobel_thresh(gray, (30, 255))

    zeros = np.zeros_like(gray)
    color_binary = np.dstack((S_filter, Sobel_filter, zeros)) * 255

    combined_binary = np.zeros_like(gray)
    combined_binary[(S_filter == 1) | (Sobel_filter == 1)] = 1

    return (combined_binary, color_binary)

def make_binary_v1(img):
    #start = time.time()
    img = np.copy(img)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    #img[:,:,0] = G_CLACHE.apply(img[:,:,0])
    img[:,:,1] = G_CLACHE.apply(img[:,:,1])
    #img[:,:,2] = G_CLACHE.apply(img[:,:,2])
    #blur5 = cv2.GaussianBlur(img,(5,5),0)
    #img = cv2.GaussianBlur(blur5,(3,3),0)

    #gray = cv2.Canny(img,150,200)

    #gray = G_CLACHE.apply(gray)

    # get separate channels
    #R_filtered = channel_select(img[:,:,0], (160, 255))
    #G_filtered = channel_select(img[:,:,1], (160, 255))
    #B_filtered = channel_select(img[:,:,2], (0, 130))
    #yellow = R_filtered & G_filtered & B_filtered

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    gray = grayscale(img)

    S_channel = hls[:,:,2]
    S_channel = G_CLACHE.apply(S_channel)
    
    S_filter = channel_select(S_channel, (220, 255))
    Sobel_filter = sobel_thresh(gray, (30, 255))
    
    zeros = np.zeros_like(gray)
    #zeros = cv2.Canny(gray, 100, 200)
    color_binary = np.dstack((S_filter, Sobel_filter,  zeros)) * 255
    
    combined_binary = np.zeros_like(gray)
    combined_binary[(S_filter == 1) | (Sobel_filter == 1)] = 1
    
    #print(time.time() - start)
    '''color_binary[:,:,0] = combined_binary
    color_binary[:,:,1] = combined_binary
    color_binary[:,:,2] = combined_binary'''

    return (combined_binary, color_binary)

def calc_persp_matr(src_corners, dst_corners):
    src = np.float32(src_corners)
    dst = np.float32(dst_corners)
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    return m, m_inv

G_SRC_CORNERS = [[220, G_H], # left bottom
                 [580, 460], # left top
                 [700, 460], # right top
                 [1120, G_H]] # right bottom
          
G_DST_CORNERS = [[280, G_H],
                 [280, 0],
                 [1000, 0],
                 [1000, G_H]]

G_M, G_M_inv = calc_persp_matr(G_SRC_CORNERS, G_DST_CORNERS)

def perspective_transform(img, M, debug = False):  
    warped = cv2.warpPerspective(img,
                                 M,
                                 (img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_LINEAR)

    return warped

def find_lane_pixels(binary_warped, vis):
    # Visualisation
    out_img = None
    if vis:
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 150

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    left_off = 0
    right_off = 0

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        ##Find the four below boundaries of the window
        win_xleft_low = leftx_current  - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        if vis:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2) 
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2) 
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If found > minpix pixels, recenter next window
        # (`right` or `leftx_current`) on their mean position
        if len(good_left_inds) > minpix:
            leftx_mean = np.int(np.mean(nonzerox[good_left_inds]))
            leftx_current = leftx_mean
            
        if len(good_right_inds) > minpix:
            rightx_mean = np.int(np.mean(nonzerox[good_right_inds]))
            rightx_current = rightx_mean

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped, vis):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped, vis)

    # Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = [1, 1, 1]
    right_fit = [1, 1, 1]
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
    except:
        print("   empty left")
    try:
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        print("    empty right")
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    if vis:
        # Colorize alne pixels
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        
        # Draw polynom approximation
        cv2.polylines(out_img,
                      [np.array([left_fitx, ploty], dtype=np.int32).T],
                      False,
                      (255,255,0),
                      thickness=2)
        cv2.polylines(out_img,
                      [np.array([right_fitx, ploty], dtype=np.int32).T],
                      False,
                      (255,255,0),
                      thickness=2)
        

    return ploty, left_fit, right_fit, left_fitx, right_fitx, out_img

#
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
#
def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1,a2, b1,b2):
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

# find k, b for x = ky + b
def fit_line(x_dots, y_dots):
    A = np.vstack([y_dots, np.ones(len(y_dots))]).T     
    k, b = np.linalg.lstsq(A, x_dots)[0]
    return (k, b)

def find_linear_lines(left_dots, right_dots, dbg_img):
    left_x = np.array(left_dots[0].T[0])
    left_y = np.array(left_dots[0].T[1])
    right_x = np.array(right_dots[0].T[0])
    right_y = np.array(right_dots[0].T[1])

    selected_left = ((left_x >=0) & (left_x <= G_W - 1) & (left_y >= 550)).nonzero()[0]
    selected_right = ((right_x >=0) & (right_x <= G_W - 1) & (right_y >= 550)).nonzero()[0]

    (lk, lb) = fit_line(left_x[selected_left], left_y[selected_left])
    (rk, rb) = fit_line(right_x[selected_right], right_y[selected_right])
    
    if dbg_img is not None:        
        dbg_img[left_y[selected_left].astype(int), left_x[selected_left].astype(int)] = [255, 0, 0]
        dbg_img[right_y[selected_right].astype(int), right_x[selected_right].astype(int)] = [0, 0, 255]
        
        y_top, y_bot = 0, 720
        cv2.line(dbg_img, (int(lk * y_bot + lb), y_bot), (int(lk * y_top + lb), y_top), [0, 255, 255], 2)
        cv2.line(dbg_img, (int(rk * y_bot + rb), y_bot), (int(rk * y_top + rb), y_top), [0, 255, 255], 2)
    
    return (lk, lb), (rk, rb)

def find_persp_rectangle(l_line, r_line, rect_top_width, dbg_img = None):    
    # find cross point
    y_top, y_bot = 0, 720
    lk, lb = l_line
    rk, rb = r_line
    lp1 = np.array([lk * y_bot + lb, y_bot])
    lp2 = np.array([lk * y_top + lb, y_top])
    rp1 = np.array([rk * y_bot + rb, y_bot])
    rp2 = np.array([rk * y_top + rb, y_top])
    
    cp = seg_intersect(lp1, lp2, rp1, rp2)
    rect_bot_width = rp1[0] - lp1[0]
    H = y_bot - cp[1]
    L = cp[0] - lp1[0]
    R = rp1[0] - cp[0]

    if (cp[0] >= 0 and cp[0] <= G_W
        and lp1[0] >= 0 and lp1[0] <= G_W
        and rp1[0] >= 0 and rp1[0] <= G_W
        and rect_bot_width > 0 #and L > 0 and R > 0
        and rect_top_width < rect_bot_width):
        
        rect_height =  H * rect_top_width / rect_bot_width
        
        rect = [lp1, np.array([cp[0] - L * rect_top_width / rect_bot_width, cp[1] + rect_height]),
                np.array([cp[0] + R * rect_top_width / rect_bot_width, cp[1] + rect_height]), rp1]
        
        if dbg_img is not None:
            cv2.polylines(dbg_img,
                          [np.array(rect, dtype=np.int32).reshape((-1, 1, 2))],
                          True, (255, 0, 0), thickness = 2)
        
        return rect
           
    return None

def fit_poly(leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each with np.polyfit()
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit
    
def find_poly_values(left_fit, right_fit, img_shape):
    # Generate x and y values for plotting
    ploty = np.linspace(0, G_H - 1, G_H)
    # Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 70

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### Set the area of search based on activated x-values
    ### within the +/- margin of our polynomial function
    ### Hint: consider the window areas for the similarly named variables
    ### in the previous quiz, but change the windows to our new search area
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
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    min_found = 100
    new_left_fit, new_right_fit = None, None
    if len(leftx) < min_found or len(rightx) < min_found:
        new_left_fit = left_fit
        new_right_fit = right_fit
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        return ploty, new_left_fit, new_right_fit, None, None, out_img  
    
    # Fit new polynomials
    new_left_fit, new_right_fit = fit_poly(leftx, lefty, rightx, righty)
    
    if (lefty < 360).sum() <= 10:
        A = np.vstack([lefty, np.ones(len(lefty))]).T     
        k, b = np.linalg.lstsq(A, leftx)[0]
        new_left_fit[0] = 0
        new_left_fit[1] = k
        new_left_fit[2] = b
        
    if (righty < 360).sum() <= 10:
        A = np.vstack([righty, np.ones(len(righty))]).T     
        k, b = np.linalg.lstsq(A, rightx)[0]
        new_right_fit[0] = 0
        new_right_fit[1] = k
        new_right_fit[2] = b
    
    left_fitx, right_fitx, ploty = find_poly_values(left_fit, right_fit,
                                                    binary_warped.shape)
    new_left_fitx, new_right_fitx, ploty = find_poly_values(new_left_fit, new_right_fit,
                                                            binary_warped.shape)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    cv2.polylines(out_img,
                  [np.array([new_left_fitx, ploty], dtype=np.int32).T],
                  False,
                  (255,255,0),
                  thickness=2)
    cv2.polylines(out_img,
                  [np.array([new_right_fitx, ploty], dtype=np.int32).T],
                  False,
                  (255,255,0),
                  thickness=2)
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
       
    return ploty, new_left_fit, new_right_fit, new_left_fitx, new_right_fitx, result

def measure_curvature_real(ploty, left_fitx, right_fitx, M_inv):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Do the calculation of R_curve (radius of curvature) 
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # To calculate car position
    # 1) translate line coordinates back into perspective view
    # 2) calculate xm_per_pix for bottom of perspective view as we know the width of a lane
    # 2) calculate difference between image center and arithmetcial mean of lines coordinates and multiply it by xm_per_pix
    linesx = np.array([left_fitx[-1], right_fitx[-1]])
    linesy = np.array([720, 720])
    orig_line_dots = cv2.perspectiveTransform(np.array([np.array([linesx, linesy], dtype=np.float32).T]), M_inv)
    
    xm_per_pix = 3.7 / np.abs(orig_line_dots[0][0][0] - orig_line_dots[0][1][0])
    pos = (img_size[0]/2 - (orig_line_dots[0][0][0] + orig_line_dots[0][1][0])/2) * xm_per_pix
    
    return left_curverad, right_curverad, pos

def draw_lane(undist, warped, left_fitx, right_fitx, ploty, M_inv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (G_W, G_H)) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    return result

import cv2

def draw_text(img, l_curverad, r_curverad, relative_pos,
              is_found = None, left_fit = None, right_fit = None):
    font = cv2.FONT_HERSHEY_DUPLEX
    curve = ((l_curverad + r_curverad) / 2)
    rad = '%.2f m' % curve if curve is not None else '-'
    pos = '%.2f m' % relative_pos if relative_pos is not None else '-'
    cv2.putText(img, 'Radius: ' + rad, (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, 'Position: ' + pos, (50, 90), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return img

def make_binary(img):
    img = np.copy(img)

    # get separate channels
    R_channel = img[:,:,0]
    G_channel = img[:,:,1]
    B_channel = img[:,:,2]
    diff_RG_abs = abs(np.int32(R_channel) - np.int32(G_channel))
    diff_GB_abs = abs(np.int32(B_channel) - np.int32(G_channel))
    diff_RB_abs = abs(np.int32(B_channel) - np.int32(R_channel))
    gray = grayscale(img)
    wl = np.uint8(R_channel*0.299 + G_channel*0.587 + B_channel*0.1170)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S_channel = hls[:,:,2]

    gray_cl = G_CLACHE.apply(gray)
    R_channel_cl = G_CLACHE.apply(img[:,:,0])
    G_channel_cl = G_CLACHE.apply(img[:,:,1])
    B_channel_cl = G_CLACHE.apply(img[:,:,2])

    #
    # S filter
    #
    # Gray mask
    gray_reg = np.zeros_like(gray)
    gray_reg[(diff_RG_abs <= 25) & (diff_GB_abs <= 25) & (diff_RB_abs <= 25)] = 1
    gray_reg_er = cv2.erode(gray_reg, np.ones((5,5)), iterations = 2)
    # Filter
    S_filter = sobel_thresh(S_channel,(30, 255))
    S_filter[(gray_reg_er == 1)] = 0

    #
    # Sobel filters
    #
    # Gray with removed white mask
    white_mask = channel_select(gray, (180, 255))
    white_mask = cv2.dilate(white_mask, np.ones((5,5)), iterations = 1)
    gray_nowhite_reg = np.copy(gray_reg_er)
    gray_nowhite_reg[(white_mask == 1)] = 0
    # Common filter
    sobel_x = sobel_thresh(gray, (30, 255))
    sobel_x[(gray_nowhite_reg == 1)] = 0
    # Shadow filter
    sobel_x_low = sobel_thresh(gray, (15, 255))
    bright_mask = channel_select(wl, (100, 255))
    bright_mask_dl = cv2.dilate(bright_mask, np.ones((5,5)), iterations = 1)
    sobel_shadows = np.copy(sobel_x_low)
    sobel_shadows[(bright_mask_dl == 1)] = 0
    # Combine common and shadow filters
    Sobel_filter = sobel_x | sobel_shadows

    #
    # Color filter
    #
    # Color mask
    color_reg = np.zeros_like(gray)
    color_reg[(diff_RG_abs > 15) & (diff_GB_abs > 15) & (diff_RB_abs > 15)] = 1
    color_reg_er = cv2.erode(color_reg, np.ones((5,5)), iterations = 2)
    # Yellow filter
    # (Good detection of yellow lane on challenge video)
    R_filtered = channel_select(img[:,:,0], (160, 255))
    G_filtered = channel_select(img[:,:,1], (160, 255))
    B_filtered = channel_select(img[:,:,2], (0, 130))
    yellow = R_filtered & G_filtered & B_filtered
    # White filter
    white = channel_select(gray, (210, 255))
    low_light_reg = channel_select(wl, (0, 150))
    low_light_reg_dl = cv2.dilate(low_light_reg, np.ones((5,5)), iterations = 1)
    white &= low_light_reg_dl
    # Light filter
    sobel_light = sobel_thresh(wl, (25, 255))
    lower_light_reg = channel_select(wl, (0, 120))
    lower_light_reg_dl = cv2.dilate(lower_light_reg, np.ones((5,5)), iterations = 1)
    sobel_light[(lower_light_reg_dl == 1)] = 0

    Color_filter = yellow | white | sobel_light
    Color_filter[(color_reg_er == 1)] = 0

    #
    # Output
    #
    zeros = np.zeros_like(gray)
    color_binary = np.dstack((S_filter, Sobel_filter, Color_filter)) * 255
    combined_binary = np.zeros_like(gray)
    combined_binary[(S_filter == 1) | (Sobel_filter == 1) | (Color_filter == 1)] = 1
    return (combined_binary, color_binary)


def gimp_to_opencv_hsv(*hsv):
    """
    I use GIMP to visualize colors. This is a simple
    GIMP => CV2 HSV format converter.
    """
    return (hsv[0] / 2, hsv[1] / 100 * 255, hsv[2] / 100 * 255)
# White and yellow color thresholds for lines masking.
# Optional "kernel" key is used for additional morphology
'''
WHITE_LINES = { 'low_th': gimp_to_opencv_hsv(0, 0, 80),
                'high_th': gimp_to_opencv_hsv(359, 10, 100) }

YELLOW_LINES = { 'low_th': gimp_to_opencv_hsv(35, 20, 20),
                 'high_th': gimp_to_opencv_hsv(90, 100, 100),
                 'kernel': np.ones((3,3),np.uint64)
                }
'''
YELLOW_LINES = { 'low_th': (225, 180, 0),
                'high_th': (255, 255, 170) }
WHITE_LINES = { 'low_th': (100, 100, 200),
                 'high_th': (255, 255, 255)
}
def get_lane_lines_mask(hsv_image, colors):
    """
    Image binarization using a list of colors. The result is a binary mask
    which is a sum of binary masks for each color.
    """
    masks = []
    for color in colors:
        if 'low_th' in color and 'high_th' in color:
            mask = cv2.inRange(hsv_image, color['low_th'], color['high_th'])
            if 'kernel' in color:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, color['kernel'])
            #cv2.imwrite('mask.jpg',
            #    cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
            masks.append(mask)
        else: raise Exception('High or low threshold values missing')
    if masks:
        return cv2.add(*masks)

def draw_binary_mask(binary_mask, img):
    if len(binary_mask.shape) != 2:
        raise Exception('binary_mask: not a 1-channel mask. Shape: {}'.format(str(binary_mask.shape)))
    masked_image = np.zeros_like(img)
    for i in range(3):
        masked_image[:,:,i] = binary_mask.copy()
    return masked_image

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def make_binary_v2(img):
    image = np.copy(img)
    #hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    binary_mask = get_lane_lines_mask(image, [WHITE_LINES, YELLOW_LINES])
    masked_image = draw_binary_mask(binary_mask, image)
    #blank_image = np.zeros_like(masked_image)
    edges_mask = canny(masked_image, 240, 400)
    '''edges_mask = masked_image
    image[:,:,0] = edges_mask
    image[:,:,1] = edges_mask
    image[:,:,2] = edges_mask'''
    edges_mask[edges_mask == 255] = 1
    #masked_image[masked_image == 255] = 1
    return (edges_mask, image)
