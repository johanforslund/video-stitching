import numpy as np
import random
from scipy.spatial import distance

def find_homography(kp1, kp2, N):
    best_inlier_count = 0
    best_H = None
    best_inliers1 = None
    best_inliers2 = None

    for i in range(N):
        random_index = random.sample(range(kp1.shape[1]), 4)

        kp1_rand = kp1[:, random_index]
        kp2_rand = kp2[:, random_index]
        
        T = create_similiarity_mat(kp1_rand)

        kp1_rand_norm = make_homogenous(kp1_rand)
        kp2_rand_norm = make_homogenous(kp2_rand)

        kp1_rand_norm = np.matmul(T, kp1_rand_norm)
        kp2_rand_norm = np.matmul(T, kp2_rand_norm)

        A = DLT(kp1_rand_norm, kp2_rand_norm)

        U, S, V_t = np.linalg.svd(A)
        V = V_t.T
        
        h_hat = V[:, -1]
        H_hat = h_hat.reshape((3, 3))

        H = np.array(np.linalg.inv(T) @ H_hat @ T)

        kp1_norm = make_homogenous(kp1)
        kp2_norm = make_homogenous(kp2)
        
        pts1 = np.matmul(H, kp1_norm)
        pts1 = pts1 / pts1[-1, :]
        pts2 = kp2_norm
        
        d = np.sqrt(np.sum((pts1 - pts2)**2, axis=0))

        thresh = 3
        inlier_count = 0

        inliers1 = []
        inliers2 = []

        for i, distance in enumerate(d):
            if distance < thresh:
                inlier_count = inlier_count + 1
                inliers1.append(kp1[:, i])
                inliers2.append(kp2[:, i])

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_H = H
            best_inliers1 = np.array(inliers1)
            best_inliers2 = np.array(inliers2)

    print(best_inlier_count / d.shape[0])
        
    return best_H

def make_homogenous(pts):
    col = np.ones((1, pts.shape[1]))
    res = np.append(pts, col, axis=0)
    return res

def DLT(pts1, pts2):
    A = np.zeros((8, 9))

    for i in range(4):
        x1 = pts1[0, i]
        y1 = pts1[1, i]
        x2 = pts2[0, i]
        y2 = pts2[1, i]
        
        p_i = np.zeros([2, 9])
        p_i[0, :] = np.array([-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2])
        p_i[1, :] = np.array([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])

        row = i*2
        A[row:row+2, :] = p_i

    return A

def create_similiarity_mat(pts):
    x = pts[0, :]
    y = pts[1, :]

    x_mean = np.sum(x) / len(x)
    y_mean = np.sum(y) / len(y)

    s = np.sqrt(2) * 4 / np.sum(np.sqrt((x - x_mean)**2 + (y - y_mean)**2))
    T = s * np.matrix([[1, 0, -x_mean], [0, 1, -y_mean], [0, 0, 1/s]])

    return T