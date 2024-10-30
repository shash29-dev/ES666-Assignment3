import pdb
import glob
import cv2
import os
import numpy as np
import random

class PanaromaStitcher:
    def _init_(self):
        pass
    
    def make_panaroma_for_images_in(self, path):
        # Get sorted list of image paths
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print(f'Found {len(all_images)} images for stitching.')

        # Load the first image
        base_img = cv2.imread(all_images[0])
        homography_matrices = []

        # Iterate through remaining images and stitch
        for img_path in all_images[1:]:
            next_img = cv2.imread(img_path)

            # Downscale to avoid memory issues
            scale_percent = 30
            base_img = cv2.resize(base_img, None, fx=scale_percent / 100, fy=scale_percent / 100)
            next_img = cv2.resize(next_img, None, fx=scale_percent / 100, fy=scale_percent / 100)

            # Stitch images and compute homography
            stitched_img, H = self.stitch_images(base_img, next_img)
            homography_matrices.append(H)

            # Update base_img for the next iteration
            base_img = stitched_img

        return base_img, homography_matrices
    
    def stitch_images(self, left_img, right_img):
        key_points1, descriptor1, key_points2, descriptor2 = self.get_keypoints(left_img, right_img)
        good_matches = self.match_keypoints(key_points1, key_points2, descriptor1, descriptor2)
        final_H = self.ransac(good_matches)

        # Transform dimensions for panorama image
        rows1, cols1 = right_img.shape[:2]
        rows2, cols2 = left_img.shape[:2]

        points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
        points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
        points2 = cv2.perspectiveTransform(points, final_H)
        list_of_points = np.concatenate((points1, points2), axis=0)

        [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

        # Translation matrix
        H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]) @ final_H

        # Warp the left image
        stitched_img = cv2.warpPerspective(left_img, H_translation, (x_max - x_min, y_max - y_min))
        stitched_img[-y_min:rows1 - y_min, -x_min:cols1 - x_min] = right_img

        return stitched_img, final_H

    def get_keypoints(self, left_img, right_img):
        l_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        r_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        key_points1, descriptor1 = sift.detectAndCompute(l_img, None)
        key_points2, descriptor2 = sift.detectAndCompute(r_img, None)

        return key_points1, descriptor1, key_points2, descriptor2

    def match_keypoints(self, key_points1, key_points2, descriptor1, descriptor2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptor1, descriptor2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                left_pt = key_points1[m.queryIdx].pt
                right_pt = key_points2[m.trainIdx].pt
                good_matches.append([left_pt[0], left_pt[1], right_pt[0], right_pt[1]])

        return good_matches

    def homography(self, points):
        A = []
        for pt in points:
            x, y = pt[0], pt[1]
            X, Y = pt[2], pt[3]
            A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
            A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])

        A = np.array(A)
        _, _, vh = np.linalg.svd(A)
        H = (vh[-1, :].reshape(3, 3))
        H = H / H[2, 2]
        return H

    def ransac(self, good_pts):
        best_inliers = []
        final_H = []
        threshold = 4

        for _ in range(100):
            random_pts = random.choices(good_pts, k=4)
            H = self.homography(random_pts)
            inliers = []

            for pt in good_pts:
                p = np.array([pt[0], pt[1], 1]).reshape(3, 1)
                p_1 = np.array([pt[2], pt[3], 1]).reshape(3, 1)
                Hp = np.dot(H, p)
                Hp = Hp / Hp[2]
                dist = np.linalg.norm(p_1 - Hp)

                if dist < threshold:
                    inliers.append(pt)

            if len(inliers) > len(best_inliers):
                best_inliers, final_H = inliers, H

        return final_H
    


######