#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 23:11:10 2019

@author: chunqi
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from moviepy.editor import VideoFileClip

class advancedLaneFinding:
    def __init__(self):
        self.left_fit = np.array([0,0,0])
        self.right_fit = np.array([0,0,0])
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        # intialize image shape
        self.imageSize = (0,0)
        self.firstFrame = 1
        return
        
    def cameraCalibration(self,boardWidth=9,boardHeight=6):
        # function to find object and image points for camera calibration
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((boardHeight*boardWidth,3), np.float32)
        objp[:,:2] = np.mgrid[0:boardWidth,0:boardHeight].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('../camera_cal/calibration*.jpg')

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                # uncomment to save figure               
                # cv2.imshow('image',img)
                # cv2.imwrite('../output_images/chessBoard/'+fname[14:], img)
                # cv2.waitKey(500)
        
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)
   
    def undistort(self,img):
        # function to create undistort images
        # Use cv2.calibrateCamera() and cv2.undistort()
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img.shape[1:], None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        return undist
    
    def undistortTest(self,path):
        # function to test self.undistort with provided test images
        images = glob.glob(path)
        for fname in images:
            img = cv2.imread(fname)
            undist = self.undistort(img)
            # uncomment to save figure
            # cv2.imwrite('../output_images/undistort/'+fname[15:], undist)
            
    def thresholdImage(self,img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        # function to created thresholded binary image
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # Stack each channel
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        
        # Combine x gradient and color channel
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
        return color_binary, combined_binary

    def thresholdTest(self,path):
        # function to test self.thresholdImage with provided test images
        images = glob.glob(path)
        for fname in images:
            img = cv2.imread(fname)
            color_binary, combined_binary = self.thresholdImage(img)
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            ax1.set_title('Stacked thresholds')
            ax1.imshow(color_binary)

            ax2.set_title('Combined S channel and gradient thresholds')
            ax2.imshow(combined_binary, cmap='gray')
            plt.imsave('../output_images/threshold/v2'+fname[-9:], combined_binary, cmap=cm.gray)
            # plt.savefig('../output_images/threshold/'+fname[-9:])
            
    def birdsEye(self, img, s=[[592,450],[688,450],[1120,720],[200,720]],d=[[240,0],[1040,0],[1040,720],[240,720]]):
        # function to warp the images to birdeye view
        self.imageSize = (img.shape[1], img.shape[0])
        src = np.float32(s)
        dst = np.float32(d)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, self.imageSize)
        return warped
        
    def birdsEyeTest(self,path):
        # function to test self.birdsEye with provided test images
        images = glob.glob(path)
        for fname in images:
            img = cv2.imread(fname)
            warped = self.birdsEye(img)
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
            f.tight_layout()
            ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax1.set_title('Undistorted Image', fontsize=20)
            ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            ax2.set_title('Undistorted and Warped Image', fontsize=20)
            plt.savefig('../output_images/birdeye/'+fname[-9:])
            cv2.imwrite('../output_images/birdeye/v2'+fname[-9:], warped)
    
    def find_lane_pixels(self,binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[self.imageSize[1]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
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
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(self.imageSize[1]//nwindows)
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
    
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.imageSize[1] - (window+1)*window_height
            win_y_high = self.imageSize[1] - window*window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin # Update this
            win_xleft_high = leftx_current + margin  # Update this
            win_xright_low = rightx_current - margin  # Update this
            win_xright_high = rightx_current + margin  # Update this
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                          (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
                          (win_xright_high,win_y_high),(0,255,0), 2) 
            
            ### TO-DO: Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            ### TO-DO: If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###\
            # Remove this when you add your function
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                    
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

    def fit_firstPoly(self, binary_warped):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)
        
        ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, self.imageSize[1]-1, self.imageSize[1])
        try:
            left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        self.firstFrame = 0
        # Plots the left and right polynomials on the lane lines
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        return out_img
    
    def fit_poly(self, leftx, lefty, rightx, righty):
        ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, self.imageSize[1]-1, self.imageSize[1])
        ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        return left_fitx, right_fitx, ploty
 
    def search_around_poly(self, binary_warped):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        margin = 100
        
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
    
        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
                                       self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + 
                                       self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
                                        self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + 
                                        self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))
    
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fitx, right_fitx, ploty = self.fit_poly(leftx, lefty, rightx, righty)
        
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
        # Plot the polynomial lines onto the image
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##    
        return left_fitx, right_fitx, ploty,result  
          
    def measure_curvature_real(self, leftx, rightx, ploty):

        # Calculates the curvature of polynomial functions in meters.        
        left_fit_cr = np.polyfit(ploty*self.ym_per_pix, leftx*self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*self.ym_per_pix, rightx*self.xm_per_pix, 2)
               
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
      
        ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*self.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*self.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        return left_curverad, right_curverad
    
    def measure_vehicle_position(self, left_detected, right_detected):
        # function to compute the vehicle offset
        position = (left_detected +right_detected)/2
        offset = (self.imageSize[0]/2 - position) * self.xm_per_pix
        return offset
    
    def inverse_perspective(self, undist, warped, left_fitx, right_fitx, ploty, d=[[592,450],[688,450],[1120,720],[200,720]],s=[[240,0],[1040,0],[1040,720],[240,720]]):
        # funtion to inverse transform
        out_img = np.dstack((warped, warped, warped))*255
        left_line_window = np.array(np.transpose(np.vstack([left_fitx, ploty])))
        right_line_window = np.array(np.flipud(np.transpose(np.vstack([right_fitx, ploty]))))
        line_points = np.vstack((left_line_window, right_line_window))
        cv2.fillPoly(out_img, np.int_([line_points]), [0,255, 0])
        
        src = np.float32(s)
        dst = np.float32(d)
        Minv = cv2.getPerspectiveTransform(src, dst)
        unwarped = cv2.warpPerspective(out_img, Minv, self.imageSize, flags=cv2.INTER_LINEAR)
        result_img = cv2.addWeighted(undist, 1, unwarped, 0.3, 0)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        return result_img
    
    def pipe_line(self,img):
        self.cameraCalibration() 
        undist = self.undistort(img)
        color_binary, combined_binary = self.thresholdImage(undist)
        warped = self.birdsEye(combined_binary)
        if self.firstFrame:
            self.fit_firstPoly(warped)
        left_fitx, right_fitx, ploty, result  = self.search_around_poly(warped)
        left_curverad, right_curverad= self.measure_curvature_real(left_fitx, right_fitx, ploty)
        offset = self.measure_vehicle_position(left_fitx[-1], right_fitx[-1])
        result_img = self.inverse_perspective(undist, warped, left_fitx, right_fitx, ploty)
        return result_img
         
    
if __name__ == '__main__':
      test1 = advancedLaneFinding()
      '''
      test1.cameraCalibration() 
      img = cv2.imread('../test_images/test1.jpg')
      undist = test1.undistort(img)
      color_binary, combined_binary = test1.thresholdImage(undist)
      warped = test1.birdsEye(combined_binary)
      test1.fit_firstPoly(warped)
      left_fitx, right_fitx, ploty, result  = test1.search_around_poly(warped)
      left_curverad, right_curverad= test1.measure_curvature_real(left_fitx, right_fitx, ploty)
      offset = test1.measure_vehicle_position(left_fitx[-1], right_fitx[-1])
      test1.inverse_perspective(warped, left_fitx, right_fitx, ploty)
      '''
      
      output = 'project_video_output.mp4'
      clip1 = VideoFileClip("../project_video.mp4")
      white_clip = clip1.fl_image(test1.pipe_line) 
      white_clip.write_videofile(output, audio=False)
      #NOTE: this function expects color images!!

      # plt.imshow(img)
      #test1.undistortTest('../test_images/test*.jpg')
      #test1.undistortTest('../camera_cal/calibration*.jpg')
      #test1.thresholdTest('../output_images/undistort/test*.jpg')
      #test1.birdsEyeTest('../output_images/threshold/v2test*.jpg')
      