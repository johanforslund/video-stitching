import cv2
import numpy as np
from KNN import KNN
import matplotlib.pyplot as plt
from random import randrange

img_right = cv2.imread('Resources/right.jpg')
img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
img_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

img_left = cv2.imread('Resources/left.jpg')
img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
img_l = cv2.cvtColor(img_left,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img_r, None)
kp2, des2 = sift.detectAndCompute(img_l, None)

tmp_des1 = des1[0:10000, :]
tmp_des2 = des2[0:10000, :]

knn_solver = KNN(tmp_des1, tmp_des2, 2)
matches = knn_solver.solve()
print(len(matches))
