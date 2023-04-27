import cv2 as cv
import numpy as np
import os
os.chdir('./images')

my_img1 = cv.imread("Q2imageA.png") #reading image A
my_img2 = cv.imread("Q2imageB.png") #reading image B

#Using ORB detector to extract the keypoints
#creating ORB Detector
orb = cv.ORB_create(nfeatures = 1000) #by default it takes 500 as max no. of features

key_points_1, descriptors_1 = orb.detectAndCompute(my_img1, None)
key_points_2, descriptors_2 = orb.detectAndCompute(my_img2, None)

#Now, we match the features common to both images
#Create a bf(Brute Force) Matcher
bf = cv.BFMatcher_create(cv.NORM_HAMMING) #cv.NORM_HAMMING is used for distance measurement

#compare and match descriptors from image 1 with image 2
matching_points = bf.knnMatch(descriptors_1, descriptors_2, k=2)  #two nearnest neighbours

#Finding out good matches, i.e., getting rid of point which aren't distinct enough
good_matching_points = []
for m,n in matching_points:
    if m.distance < 0.8 * (n.distance):
        good_matching_points.append(m)

#Calculate the Homography matrix between two images
#first we set a minimum match count
minimum_match_count = 10 #need atlest 10 matches to connect two images
if (len(good_matching_points) > minimum_match_count):
    # Convert keypoints to an argument for findHomography
    # extracting location of matched keypoints
    source_points = np.float32([key_points_1[m.queryIdx].pt for m in good_matching_points]).reshape(-1,1,2)
    destination_points = np.float32([key_points_2[m.trainIdx].pt for m in good_matching_points]).reshape(-1,1,2)

    #find the homography between the source points and destination points
    #Using the RANSAC algorithm to calculate the homography matrix
    H_Matrix, _ = cv.findHomography(source_points, destination_points, cv.RANSAC, 5.0)

    #warping the images into same perspective
    #finding the width and height of each image
    row_1, col_1 = my_img1.shape[:2]
    row_2, col_2 = my_img2.shape[:2]

    coordinates_img_1 = np.float32([[0,0], [0, row_1],[col_1, row_1], [col_1, 0]]).reshape(-1, 1, 2)
    coordinates_img_2_to_transform = np.float32([[0,0], [0,row_2], [col_2,row_2], [col_2,0]]).reshape(-1,1,2)

    #changing perspective for image 1 to match with image 2 to align them properly
    coordinates_img_2 = cv.perspectiveTransform(coordinates_img_2_to_transform, H_Matrix)

    #joining both the coordinates arrays along same axis
    coordinates = np.concatenate((coordinates_img_1, coordinates_img_2), axis = 0) 
    
    #getting the minimum and maximum values in each coordinate, x and y
    [x_min, y_min] = np.int32(coordinates.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(coordinates.max(axis=0).ravel() + 0.5)

    #distance image needs to be moved
    translation_distance = [-x_min, -y_min]

    #translation matrix for the moved image
    H_translation = np.array([[1, 0, translation_distance[0]], [0, 1, translation_distance[1]], [0, 0, 1]])

    #warping first image to match second image
    Stitched_img = cv.warpPerspective(my_img1, H_translation.dot(H_Matrix), (x_max-x_min, y_max-y_min))
    #stitching the two images together based on translation distance
    Stitched_img[translation_distance[1]:row_1+translation_distance[1], translation_distance[0]:col_1+translation_distance[0]] = my_img2
    cv.imshow("Stitched", Stitched_img)
  
    # Using cv2.imwrite() method
    # Saving the image
    filename = 'stiched_image.jpg'
    cv.imwrite(filename, Stitched_img)
    cv.waitKey(0)

cv.destroyAllWindows()