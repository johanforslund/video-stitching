import cv2
import numpy as np
from KNN import KNN
import matplotlib.pyplot as plt
from random import randrange
from helpers import show_corresp 
from homography import find_homography

img_right = cv2.imread('Resources/right.jpg')
img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
img_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

img_left = cv2.imread('Resources/left.jpg')
img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
img_l = cv2.cvtColor(img_left,cv2.COLOR_BGR2GRAY)

# Create SIFT and extract features
sift = cv2.xfeatures2d.SIFT_create(nfeatures=10000)

# Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img_r, None)
kp2, des2 = sift.detectAndCompute(img_l, None)

# Find KNN matches and validate with ratio test
knn_solver = KNN(des1, des2, 2)
matches = knn_solver.solve()

# Extract keypoints' coordinates
valid_kp1 = []
valid_kp2 = []

for match in matches:
    valid_kp1.append(np.array(kp1[match.index_1].pt))
    valid_kp2.append(np.array(kp2[match.indices_2[0]].pt))

valid_kp1 = np.array(valid_kp1).T
valid_kp2 = np.array(valid_kp2).T

H = find_homography(valid_kp1, valid_kp2, 5000)
print(H)

# Visualize matches
'''
show_corresp(img_r, img_l, valid_kp1, valid_kp2, vertical=0)
plt.show()
'''

# Transform and output
dst = cv2.warpPerspective(img_right,H,(img_left.shape[1] + img_right.shape[1], img_left.shape[0]))     	
plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
plt.show()
plt.figure()
dst[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
plt.imshow(dst)
plt.show()
cv2.imwrite('resultant_stitched_panorama.jpg',dst)