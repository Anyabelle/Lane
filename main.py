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

from lib import undistort
from lib import calc_persp_matr
from lib import find_linear_lines
from lib import search_around_poly
from lib import fit_polynomial
from lib import make_binary_v1
from lib import make_binary_v2
from lib import make_binary
from lib import perspective_transform
from lib import fit_poly
from lib import find_persp_rectangle
from lib import find_poly_values
from lib import measure_curvature_real
from lib import draw_lane
from lib import make_binary_fast
#
# Define global constants corresponding to image dimensions
#
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

img = mpimg.imread('test_images/straight_lines1.jpg') 
undist = undistort(img)
comb_bin, colored = make_binary(undist)
G_SRC_CORNERS = [[220, G_H], # left bottom
                 [580, 460], # left top
                 [700, 460], # right top
                 [1120, G_H]] # right bottom
          
G_DST_CORNERS = [[280, G_H],
                 [280, 0],
                 [1000, 0],
                 [1000, G_H]]
G_M, G_M_inv = calc_persp_matr(G_SRC_CORNERS, G_DST_CORNERS)
warped = perspective_transform(comb_bin, G_M)

py, lf, rf, lfx, rfx, vis = fit_polynomial(warped, True)


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.detected = False  # was the line detected in the last iteration?
        self.recent_xfitted = [] # x values of the last n fits of the line
        self.recent_fit = [] # polynomial coefficients of the last n fits of the line
        self.bestx = None #average x values of the fitted line over the last n iterations
        self.best_fit = None #polynomial coefficients averaged over the last n iterations
        self.current_fit = [np.array([False])] #polynomial coefficients for the most recent fit  
        self.rad = None #radius of curvature of the line in some units
        self.allx = None #x values for detected line pixels
        self.ally = None #y values for detected line pixels
          
class ProcessSeq:
    WARPED_PIXELS_THRES = 110000
    MAX_LOST_FRAMES = 5
    HISTORY_SIZE = 6
    MIN_FRAMES_TO_AVERAGE = 3
    PERSP_RECT_TOP_MIN = 100
    PERSP_RECT_TOP_MAX = 350
    PERSP_RECT_TOP_DEF = 250
    
    def __init__(self, do_average = False, do_debug = False, def_corners = None, def_width = None):
        self.l_lane = Line()
        self.r_lane = Line()
        self.frame = 0
        self.lost_frames = self.MAX_LOST_FRAMES
        self.lane_pos = 0
        self.curv = 0
        self.persp_rect_top = self.PERSP_RECT_TOP_DEF
        if def_width is not None:
            self.persp_rect_top = def_width
        self.history_len = 0
        self.warped_pixels = 0
        self.do_average = do_average
        self.do_debug = do_debug
        lmargin, hmargin, yedge = 220, 515, 500
        self.def_corners = [[lmargin, G_H], # left bottom
                            [hmargin, yedge], # left top
                            [G_W - hmargin, yedge], # right top
                            [G_W - lmargin, G_H]] # right bottom
        if def_corners is not None:
            self.def_corners = def_corners
        self.def_M, self.def_M_inv = calc_persp_matr(self.def_corners, G_DST_CORNERS)

        self.corners = self.def_corners
        self.M, self.M_inv = self.def_M, self.def_M_inv
        
        self.vis_bin = None
    
    def decrease_view_range(self):
        self.persp_rect_top += 10
        if self.persp_rect_top > self.PERSP_RECT_TOP_MAX:
            self.persp_rect_top = self.PERSP_RECT_TOP_MAX
            
    def increase_view_range(self):
        self.persp_rect_top -= 10
        if self.persp_rect_top < self.PERSP_RECT_TOP_MIN:
            self.persp_rect_top = self.PERSP_RECT_TOP_MIN
    
    def reset_view_range(self):
        self.top_persp_rec = self.PERSP_RECT_TOP_DEF
        
    
    def fit_dots(self, left_dots, right_dots):
        leftx = left_dots[0].T[0]
        lefty = left_dots[0].T[1]
        rightx = right_dots[0].T[0]
        righty = right_dots[0].T[1]

        lf, rf = fit_poly(leftx, lefty, rightx, righty)
        lfx, rfx, py = find_poly_values(lf, rf, G_SHAPE)
        
        return lf, rf, lfx, rfx
    
    def refit_lane(self, old_M_inv, new_M, lfx, rfx):
        # Convert back to top view
        py = np.linspace(0, G_H - 1, G_H)
        left_dots = cv2.perspectiveTransform(np.array([np.array([lfx, py], dtype=np.float32).T]), old_M_inv)
        right_dots = cv2.perspectiveTransform(np.array([np.array([rfx, py], dtype=np.float32).T]), old_M_inv)

        # Convert to perspective view with new perspective transform
        left_dots_p = cv2.perspectiveTransform(left_dots, new_M)
        right_dots_p = cv2.perspectiveTransform(right_dots, new_M)

        return self.fit_dots(left_dots_p, right_dots_p)
    
    def update_history(self, old_M_inv, new_M):
        for idx in range(self.history_len):
            lf, rf, lfx, rfx = self.refit_lane(old_M_inv, new_M,
                                               self.l_lane.recent_xfitted[idx], self.r_lane.recent_xfitted[idx])
            self.l_lane.recent_fit[idx], self.r_lane.recent_fit[idx] = lf, rf
            self.l_lane.recent_xfitted[idx], self.r_lane.recent_xfitted[idx] = lfx, rfx
        
    def update_perspective(self, lf, rf, lfx, rfx):
        # Translate lines coordinates to original image
        py = np.linspace(0, G_H - 1, G_H)
        left_dots = cv2.perspectiveTransform(np.array([np.array([lfx, py], dtype=np.float32).T]), self.M_inv)
        right_dots = cv2.perspectiveTransform(np.array([np.array([rfx, py], dtype=np.float32).T]), self.M_inv)
               
        # Find linear lines approximation
        l_line, r_line = find_linear_lines(left_dots, right_dots, self.vis_bin)
        
        # Calculate new perspective matrix 
        corners = find_persp_rectangle(l_line, r_line, self.persp_rect_top)
        
        new_M, new_M_inv = None, None
        if corners:
            new_M, new_M_inv = calc_persp_matr(corners, G_DST_CORNERS)
        
        if corners is not None and new_M is not None:           
            # Update current fit
            left_dots_p = cv2.perspectiveTransform(left_dots, new_M)
            right_dots_p = cv2.perspectiveTransform(right_dots, new_M)
            lf, rf, lfx, rfx = self.fit_dots(left_dots_p, right_dots_p)
            
            if not self.is_good_lane(lf, rf, lfx, rfx):
                return False, lf, rf, lfx, rfx
            
            if self.do_debug:
                cv2.polylines(self.vis_bin,
                              [np.array(corners, dtype=np.int32).reshape((-1, 1, 2))],
                              True,
                              (255,0,0),
                              thickness=2)
                
            # Update history (as previos fits became invalid)
            self.update_history(self.M_inv, new_M)

            # Remember new perspective transform matrix
            self.M = new_M
            self.M_inv = new_M_inv
            self.corners = corners
    
        return True, lf, rf, lfx, rfx 
        
    def is_good_lane(self, left_fit, right_fit, left_fitx, right_fitx):
        if left_fitx is None:
            return False
        
        # If lane cuvature changed too agressively, detection is incorrect
        if self.history_len > 0:
            l_diff = abs(left_fit - self.l_lane.recent_fit[-1])
            r_diff = abs(right_fit - self.r_lane.recent_fit[-1])
            if (l_diff[0] > 5.0e-04 or r_diff[0] > 5.0e-04):
                return False
        
        # If lanes have too different curvature, detection is incorrect
        if (min(abs(left_fit[0]), abs(right_fit[0])) < 2.0e-04
            and abs(left_fit[0] - right_fit[0]) > 6.0e-04):
            return False
        
        if (abs(left_fit[0] - right_fit[0]) > 10.0e-04):
            return False
        
        min_dist = 550
        max_dist = 850
        
        # If distance between lines (in the bootom of image) is too big or too small,
        # lane was detected incorrectly
        dist_bot = abs(left_fitx[G_H-1] - right_fitx[G_H-1])
        if (dist_bot < min_dist or dist_bot > max_dist):
            return False
        
        return True
    
    def is_sharp_turn(self, lf, rf, lfx, rfx, threshold):
        # sharp turn
        if (abs(lf[0]) > threshold or abs(rf[0]) > threshold
            or np.any(lfx < 50)
            or np.any(rfx > 1230)):            
            return True
       
        return False
        
    def is_flat_lane(self, lf, rf):
        if (abs(lf[0]) < 2.0e-04 and abs(rf[0]) < 2.0e-04):           
            return True
        else:
            return False
    
    def reset_history(self):
        del self.l_lane.recent_xfitted[:]
        del self.r_lane.recent_xfitted[:]
        del self.l_lane.recent_fit[:]
        del self.r_lane.recent_fit[:]
        self.history_len = 0
        
    def process_good_lane(self):
        lf, rf = self.l_lane.current_fit, self.r_lane.current_fit
        lfx, rfx = self.l_lane.allx, self.r_lane.allx
        py = self.l_lane.ally
        
        is_good = True
        if self.lost_frames >= self.MAX_LOST_FRAMES:
            self.reset_view_range()
            is_good, lf, rf, lfx, rfx = self.update_perspective(lf, rf, lfx, rfx)
        elif self.is_sharp_turn(lf, rf, lfx, rfx, 3.5e-4):
            self.decrease_view_range()
            is_good, lf, rf, lfx, rfx = self.update_perspective(lf, rf, lfx, rfx)
        elif self.is_flat_lane(lf, rf):
            self.increase_view_range()
            is_good, lf, rf, lfx, rfx = self.update_perspective(lf, rf, lfx, rfx)
        
        self.l_lane.current_fit = lf
        self.r_lane.current_fit = rf
        self.l_lane.allx = lfx
        self.r_lane.allx = rfx
        
        if not is_good:
            return False 

        self.l_lane.detected = True
        self.r_lane.detected = True
        self.lost_frames = 0
        
        self.l_lane.recent_xfitted.append(lfx)
        self.r_lane.recent_xfitted.append(rfx)
        self.l_lane.recent_fit.append(lf)
        self.r_lane.recent_fit.append(rf)
        self.history_len = len(self.l_lane.recent_xfitted)
        if  self.history_len > self.HISTORY_SIZE:
            self.l_lane.recent_xfitted.pop(0)
            self.r_lane.recent_xfitted.pop(0)
            self.l_lane.recent_fit.pop(0)
            self.r_lane.recent_fit.pop(0)
            self.history_len -= 1

        # Calculate the radius of curvature in pixels for both lane lines
        left_curverad, right_curverad, pos = measure_curvature_real(py, lfx, rfx, self.M_inv)
        self.l_lane.rad = left_curverad
        self.r_lane.rad = right_curverad
        
        assert(self.history_len == len(self.l_lane.recent_xfitted))        
        return True
        
    def process_bad_lane(self):
        self.l_lane.detected = False
        self.r_lane.detected = False

        if self.history_len > 1:
            self.l_lane.recent_xfitted.pop(0)
            self.r_lane.recent_xfitted.pop(0)
            self.l_lane.recent_fit.pop(0)
            self.r_lane.recent_fit.pop(0)
            self.history_len -= 1    
        elif self.history_len == 0:
            self.l_lane.bestx = None
            self.r_lane.bestx = None            

        self.lost_frames += 1
        if self.lost_frames >= self.MAX_LOST_FRAMES:
            # Reset perspective transform matrix
            self.M, self.M_inv = self.def_M, self.def_M_inv
            self.corners = self.def_corners
            self.l_lane.bestx = None
            self.r_lane.bestx = None
            self.reset_history()
    
    def detect_lane(self, warped_bin_img):
        if self.lost_frames < self.MAX_LOST_FRAMES:
            ploty, lf, rf, lfx, rfx, vis_warped = search_around_poly(warped_bin_img,
                                                                     self.l_lane.recent_fit[-1],
                                                                     self.r_lane.recent_fit[-1])
        else:
            ploty, lf, rf, lfx, rfx, vis_warped = fit_polynomial(warped_bin_img, DBG)
        
        self.l_lane.allx = lfx
        self.r_lane.allx = rfx
        self.l_lane.current_fit = lf
        self.r_lane.current_fit = rf
        self.l_lane.ally = ploty
        self.r_lane.ally = ploty
        
        return vis_warped
    
    def draw_text_info(self, img):
        lf, rf = self.l_lane.current_fit, self.r_lane.current_fit
        
        if ((self.do_average and self.history_len >= self.MIN_FRAMES_TO_AVERAGE)
             or (not self.do_average and self.history_len >= 1)):
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
            
        txt_size_big = 0.8
        txt_size = 0.5
        txt_reg = 1
        txt_bold = 2
        
        font = cv2.FONT_HERSHEY_DUPLEX
        rad = '%.2f m' % self.curve if self.curve is not None else '-'
        pos = '%.2f m' % self.lane_pos if self.lane_pos is not None else '-'
        cv2.putText(img, 'Radius: ' + rad, (50, 50), font, txt_size_big, color, txt_bold, cv2.LINE_AA)
        cv2.putText(img, 'Position: ' + pos, (50, 90), font, txt_size_big, color, txt_bold, cv2.LINE_AA)
        
        if self.do_debug: 
            if self.l_lane.detected:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.putText(img, 'Frame: %d' % self.frame ,
                        (50, 140), font, txt_size, (0, 255, 0), txt_reg, cv2.LINE_AA)
            cv2.putText(img, 'Found: ' + str(self.l_lane.detected and self.r_lane.detected),
                        (200, 140), font, txt_size, color, txt_reg, cv2.LINE_AA)
            cv2.putText(img, 'History len: ' + str(self.history_len),
                        (340, 140), font, txt_size, (0, 255, 0), txt_reg, cv2.LINE_AA)
            cv2.putText(img, 'Frames lost: ' + str(self.lost_frames),
                        (480, 140), font, txt_size, (0, 255, 0), txt_reg, cv2.LINE_AA)
            
            cv2.putText(img, 'View range: ' + str(int(self.persp_rect_top)),
                        (50, 160), font, txt_size, (0, 255, 0), txt_reg, cv2.LINE_AA)
            cv2.putText(img, 'Warped pix: ' + str(self.warped_pixels),
                        (200, 160), font, txt_size, (0, 255, 0), txt_reg, cv2.LINE_AA)  

            if self.l_lane.allx is not None:
                cv2.putText(img, 'Width bot: ' + str(abs(self.l_lane.allx[G_H-1] - self.r_lane.allx[G_H-1])),
                            (380, 160), font, txt_size, (0, 255, 0), txt_reg, cv2.LINE_AA)

            cv2.putText(img, 'Left fit: ' + str(lf),
                        (50, 200), font, txt_size, color, txt_reg, cv2.LINE_AA)
            cv2.putText(img, 'Right fit: ' + str(rf),
                        (50, 220), font, txt_size, color, txt_reg, cv2.LINE_AA)
            cv2.putText(img, 'Diff: ' + str(np.subtract(lf, rf)),
                        (50, 240), font, txt_size, color, txt_reg, cv2.LINE_AA)
            
            cv2.putText(img, 'Best left fit: ' + str(self.l_lane.best_fit),
                        (50, 280), font, txt_size, (0, 255, 0), txt_reg, cv2.LINE_AA)
            cv2.putText(img, 'Best right fit: ' + str(self.r_lane.best_fit),
                        (50, 300), font, txt_size, (0, 255, 0), txt_reg, cv2.LINE_AA)
    
        return img
    
    def __call__(self, img, dump_images = False, base = "", output_dir = ""):
        # Undistort
        #start = time.time()
        undist = undistort(img)
        #print("dist", time.time() - start)
        # Make binary
        #start = time.time()

        #bin_img, self.vis_bin = make_binary(undist)
        bin_img, self.vis_bin = make_binary_v2(undist)
        #cv2.imwrite('222v.jpg',
        #        cv2.cvtColor(255 * bin_img, cv2.COLOR_BGR2RGB))
        
        #print("bin", time.time() - start)
        # Draw current perspective transform rectangle
        #start = time.time()
        cv2.polylines(self.vis_bin,
                      [np.array(self.corners, dtype=np.int32).reshape((-1, 1, 2))],
                      True,
                      (255,255,0),
                      thickness=2)
        #
        # Perspective transform
        #
        warped_bin_img = perspective_transform(bin_img, self.M)
        #cv2.imwrite('warp222v.jpg',
        #        cv2.cvtColor(warped_bin_img * 255, cv2.COLOR_BGR2RGB))
        self.warped_pixels = warped_bin_img.sum()
        #print("trans", time.time() - start)
        start = time.time()
        if self.warped_pixels < self.WARPED_PIXELS_THRES:
            #
            # Detect lane
            #
            vis_warped = self.detect_lane(warped_bin_img)
            
            #
            # Process lanes info
            #
            is_good = self.is_good_lane(self.l_lane.current_fit,
                                        self.r_lane.current_fit,
                                        self.l_lane.allx,
                                        self.r_lane.allx)
            if is_good:
                is_good = self.process_good_lane()
            
            if not is_good:
                self.process_bad_lane()
        
            #
            # Draw lane approximations on warped image
            #
            self.is_good = is_good and self.l_lane.allx is not None
            if is_good and self.l_lane.allx is not None:
                cv2.polylines(vis_warped,
                              [np.array([self.l_lane.allx, self.l_lane.ally], dtype=np.int32).T],
                              False, (255, 0, 0), thickness=2)
                cv2.polylines(vis_warped,
                              [np.array([self.r_lane.allx, self.r_lane.ally], dtype=np.int32).T],
                              False, (255, 0, 0), thickness=2)
        else:
            vis_warped = np.dstack((warped_bin_img, warped_bin_img, warped_bin_img)) * 255
            self.process_bad_lane()
        #print("detect", time.time() - start)
        #start = time.time()
        #
        # Calculate and draw approximations
        #       
        if ((self.do_average and self.history_len >= self.MIN_FRAMES_TO_AVERAGE)
             or (not self.do_average and self.history_len >= 1)):
                       
            self.l_lane.bestx = np.mean(self.l_lane.recent_xfitted, axis=0)
            self.r_lane.bestx = np.mean(self.r_lane.recent_xfitted, axis=0)
            self.l_lane.best_fit = np.mean(self.l_lane.recent_fit, axis=0)
            self.r_lane.best_fit = np.mean(self.r_lane.recent_fit, axis=0)
            
            # Draw lane approximation
            result = draw_lane(undist, warped, self.l_lane.bestx, self.r_lane.bestx, self.l_lane.ally, self.M_inv)
        
            # Calculate the radius of curvature in pixels for both lane lines
            '''
            if self.do_average:
                l_curve, r_curve, pos = measure_curvature_real(py, self.l_lane.bestx, self.r_lane.bestx, self.M_inv)
            else:
                l_curve, r_curve, pos = measure_curvature_real(py, self.l_lane.allx, self.r_lane.allx, self.M_inv)

            self.curve = (l_curve + r_curve) / 2
            
            self.lane_pos = pos
            '''
        else:
            result = undist
            
            self.curve = None
            self.lane_pos = None
        #print("draw", time.time() - start)
        #result = self.draw_text_info(result)
        self.frame += 1

        if self.do_debug:
            return np.concatenate((self.vis_bin, vis_warped, result), axis=1)
        else:
            return result

# Make a list of images
images = glob.glob('test_images/*.jpg')
images_count = len(images)
fig, ax = plt.subplots(int((images_count + 1) / 2), 2, figsize=(10, 25))

# Define named constants for debug and ndebug modes
DBG = True
NO_DBG = False


G_SRC_CORNERS = [[220, G_H], # left bottom
                 [580, 460], # left top
                 [700, 460], # right top
                 [1120, G_H]] # right bottom
          
G_DST_CORNERS = [[280, G_H],
                 [280, 0],
                 [1000, 0],
                 [1000, G_H]]

# Step through the list
'''
for idx, fname in enumerate(images):
    print("*************")
    #start = time.time()

    t_img = cv2.imread(fname)
    t_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB)
    
    # Get base name
    base = os.path.basename(fname)
    base = os.path.splitext(base)[0]
    
    #print("read", time.time() - start)
    #start = time.time()
    process_img = ProcessSeq(False, NO_DBG, G_SRC_CORNERS, 100)
    #print("proc1", time.time() - start)
    start = time.time()
    result = process_img(t_img, DBG, base, 'output_images')
    print("proc2", time.time() - start)
    #start = time.time()
    cv2.imwrite('output_images/res_' + base + '.jpg',
                cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    #print("write", time.time() - start)
    # plot original and final image
    row = int(idx / 2)
    col = idx % 2
    ax[row, col].set_title(fname, fontsize=15)
    ax[row, col].imshow(result)
    
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
'''

#RUN VIDEO

debug = NO_DBG
debug_suffix = "_dbg" if debug else ""
cap = cv2.VideoCapture("project_video.mp4")



#JUST PROCESS EVERY 16 FRAME

counter = 0
process_img = ProcessSeq(False, NO_DBG, G_SRC_CORNERS, 100)
result = img

while (True):
        start = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not (counter & 15) or not process_img.is_good:
            result = process_img(frame, DBG, "", "")
        else:
            result = draw_lane(undistort(frame), warped, process_img.l_lane.bestx, process_img.r_lane.bestx, process_img.l_lane.ally, process_img.M_inv)
            #result = process_img.draw_text_info(result)
        counter += 1
        # Display the resulting frame
        cv2.imshow('frame', result)
        print(1.0 / (time.time() - start))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



#SHOW REST 5 FRAMES WHILE 6th PROCESSED
'''
process_img = ProcessSeq(False, NO_DBG, G_SRC_CORNERS, 100)
from multiprocessing.pool import ThreadPool

cnt = 1
while (True):
        start = time.time()
        # Capture frame-by-frame
        frames = []
        for it in range(cnt):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cnt = 20
        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print("**", frames[-1].shape)

        pool = ThreadPool(processes=1)
        async_result = pool.apply_async(process_img, (frames[-1], DBG, "", ""))
        for frame in frames[:-1]:
            #print("&", frame.shape)
            result = draw_lane(undistort(frame), warped, process_img.l_lane.bestx, process_img.r_lane.bestx, process_img.l_lane.ally, process_img.M_inv)
            result = process_img.draw_text_info(result)
            cv2.imshow('frame', result)

        pool.close()
        #pool.join()
        #t1 = Thread(target=process_img, args=(frames[-1], DBG, "", ""))
        #t1.start()
        #result = t1.join()
        #result = process_
        # img(frames[0], DBG, "", "")
        # Display the resulting frame
        cv2.imshow('frame', async_result.get())

        print(cnt * 1.0 / (time.time() - start))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
'''


'''import skvideo.io
videogen = skvideo.io.vreader("project_video.mp4")
for frame in videogen:
        print(frame.shape)
'''
