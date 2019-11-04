######################################################
#Assignment 2
#File: prob5.py
#Author: Joshua D. Rego
#Description: Image Stitching
######################################################
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy import signal
from scipy.spatial import distance
import random

def plot(imgs, subplot_size=(1,1), figsz=(10,10), title=[None]):
    rows, columns = subplot_size #[0], img_array_size[1]
    plt.figure(figsize=figsz)
    for i in range(rows*columns):
        plt.subplot(rows, columns, i+1)
        plt.imshow(imgs[i])
        plt.title(title[i])
    plt.show()

# Read in left, center, and right images
img_a = cv2.imread('./input_images/keble_a.jpg')
img_b = cv2.imread('./input_images/keble_b.jpg')
img_c = cv2.imread('./input_images/keble_c.jpg')
# Plot Images
plot([img_a,img_b,img_c], (1,3), (15,15), ['Left Image','Center Image','Right Image'])
# Initialize ORB
orb = cv2.ORB_create()
# Find Features and descriptor for left image
kp_a = orb.detect(img_a, None)
kp_a, des_a = orb.compute(img_a, kp_a)
imga_kp = cv2.drawKeypoints(img_a, kp_a, color=(0,255,0), outImage=None)
# Find Features and descriptor for cente image
kp_b = orb.detect(img_b, None)
kp_b, des_b = orb.compute(img_b, kp_b)
imgb_kp = cv2.drawKeypoints(img_b, kp_b, color=(0,255,0), outImage=None)
# Find Features and descriptor for right image
kp_c = orb.detect(img_c, None)
kp_c, des_c = orb.compute(img_c, kp_c)
imgc_kp = cv2.drawKeypoints(img_c, kp_c, color=(0,255,0), outImage=None)
# Plot Images with Feature points
plot([imga_kp,imgb_kp,imgc_kp], (1,3), (15,15), ['Left Image','Center Image','Right Image'])

# Function to compute Homography Matrix
def computeH(correspondences):
    #loop through correspondences and create assemble matrix
    aList = []
    #For each matching correspondence
    for corr in correspondences:
        # Assign to point 1 and 2
        p1 = np.array([corr[0], corr[1], 1])
        p2 = np.array([corr[2], corr[3], 1])
        # Create a vector for x and y 
        a_y = [0, 0, 0, -p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2],
               p2[1] * p1[0], p2[1] * p1[1], p2[1] * p1[2]]
        a_x = [-p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2], 0, 0, 0,
               p2[0] * p1[0], p2[0] * p1[1], p2[0] * p1[2]]
        # Add to A list for current point
        aList.append(a_x)
        aList.append(a_y)
    # Convert A list to matrix
    matrixA = np.matrix(aList)
    #svd composition
    u, s, v = np.linalg.svd(matrixA)
    #reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))
    #normalize and now we have h
    h = (1/h[2,2]) * h
    return h

# Function to define Geometric Distance
def geometricDistance(correspondence, h):
    # Define point 1 from correspondence
    p1 = np.transpose(np.array([correspondence[0], correspondence[1],1]))
    # Estimate point 2 by multiplying Homography and point 1
    estimatep2 = np.dot(h, p1)
    # Normalize point
    estimatep2 = (1/estimatep2[0,2])*estimatep2
    # Define actual point 2
    p2 = np.transpose(np.array([correspondence[2], correspondence[3], 1]))
    # Find Error between estimate and actual point 2
    error = p2 - estimatep2
    return np.linalg.norm(error)

def RANSAC(corr, thresh):
    maxInliers = []
    finalH = None
    print(corr.shape)
    for i in range(5000):
        # Find 4 random points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        corr3 = corr[random.randrange(0, len(corr))]
        corr4 = corr[random.randrange(0, len(corr))]
        # Stack 4 points into a matrix
        randomFour = np.vstack((corr1, corr2))
        randomFour = np.vstack((randomFour, corr3))
        randomFour = np.vstack((randomFour, corr4))
        # Compute Homography from the 4 random points
        h = computeH(randomFour)
        # Initialize inliers list
        inliers = []
        # For each matching point
        for i in range(len(corr)):
            # Find error in estimated points
            d = geometricDistance(corr[i,:], h)
            # If error less than 2 
            if d < 2:
                # Add as inlier
                inliers.append(corr[i,:])
        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        print ("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))
        if len(maxInliers) > (len(corr)*thresh):
            break
    return finalH, maxInliers

def findMatching(img1, img2, kp1, kp2, des1, des2):
    # Define minimum good matches needed
    MIN_MATCH_COUNT = 10
    # Flann Matching parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6,#12,
                        key_size = 12,#20,
                        multi_probe_level = 1)#2)
    search_params = dict(checks = 50)
    # Initialize Flann Matching
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # Run Flann Matching on descriptors for 2 images
    matches = flann.knnMatch(des1, des2, k=2)
    # Initialize good matches list
    good = []
    # Use Lowe's ratio test to determine good matches
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    # If number of matches > minimum matches
    if len(good) > MIN_MATCH_COUNT:
        # Assign Source and destination points (Destination is center image)
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])#.reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])#.reshape(-1,1,2)
        # Combine points
        corr_pts = np.concatenate((src_pts,dst_pts), axis=1)
        # Run ransac to find Homography
        M, inliers = RANSAC(corr_pts, 0.6)
        # Find binary mask for plotting inlier matches
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.T.ravel().tolist()
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    # cv2.drawMatches Parameters
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    # Draw matches with inliers
    img_match = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    # Plot imag with matches
    plt.figure(figsize=(10,10))
    plt.imshow(img_match, 'gray'),plt.show()
    return M

# Find Homography for left to center images
M1 = findMatching(img_a, img_b, kp_a, kp_b, des_a, des_b)
# Find Homography for right to center images
M2 = findMatching(img_c, img_b, kp_c, kp_b, des_c, des_b)
# Fix x translations for stitching
M1[0,2] = M1[0,2]+790
M2[0,2] = M1[0,2]-750
# Warp left and right images
img_warp_l = cv2.warpPerspective(img_a, M1, (img_a.shape[1]+img_b.shape[1], img_a.shape[0]))
img_warp_r = cv2.warpPerspective(img_c, M2, (img_a.shape[1], img_a.shape[0]))
# Plot warped images
plot([img_warp_l, img_warp_r], (1,2), (15,15), ['Left Warp','Right Warp'])
# Start panorama image with left image
panorama = np.copy(img_warp_l)
# Stitch original center image
panorama[0:img_b.shape[0], img_a.shape[1]:] = img_b
# Add extra width for right image
panorama = np.concatenate((panorama,np.zeros((panorama.shape[0],img_c.shape[1],3))),axis=1)
# Stitch right image
panorama[0:img_b.shape[0], img_a.shape[1]+img_b.shape[1]-1:-1] = img_warp_r
# Crop extra black space on edges
panorama_crop = panorama[:,img_a.shape[1]//2:-img_a.shape[1]//2]
# Plot Panorama image
plot([panorama_crop / 255], figsz = (25,25), title=['Final Stitched Panorama Image'])
# Write image to file
cv2.imwrite('./output_images/panorama.jpeg', panorama_crop)