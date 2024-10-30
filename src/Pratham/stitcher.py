import cv2
import numpy as np
import random
import os

class PanaromaStitcher:
    def __init__(self):
        self.homography_matrix_list = []

    def make_panaroma_for_images_in(self, path):
        image_files = sorted([f for f in os.listdir(path) if f.endswith('.JPG')])
        if len(image_files) < 2:
            raise ValueError("At least two images are required for stitching.")

        # Read and scale down the first image
        left_img_path = os.path.join(path, image_files[0])
        left_img = cv2.imread(left_img_path)
        if left_img is None:
            raise ValueError(f"Failed to load image at {left_img_path}")

        scale_percent = 30
        left_img = cv2.resize(left_img, None, fx=scale_percent / 100, fy=scale_percent / 100)
        result_img = left_img

        # Process each subsequent image and stitch
        for img_file in image_files[1:]:
            right_img_path = os.path.join(path, img_file)
            right_img = cv2.imread(right_img_path)
            if right_img is None:
                print(f"Warning: Failed to load image at {right_img_path}, skipping this image.")
                continue

            right_img = cv2.resize(right_img, None, fx=scale_percent / 100, fy=scale_percent / 100)
            result_img, H = self.stitch_images(result_img, right_img)
            self.homography_matrix_list.append(H)

        return result_img, self.homography_matrix_list

    def stitch_images(self, left_img, right_img):
        key_points1, descriptor1, key_points2, descriptor2 = self.get_keypoint(left_img, right_img)
        good_matches = self.match_keypoint(key_points1, key_points2, descriptor1, descriptor2)
        final_H = self.ransac(good_matches)

        rows1, cols1 = right_img.shape[:2]
        rows2, cols2 = left_img.shape[:2]

        points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
        points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
        points2 = cv2.perspectiveTransform(points, final_H)
        list_of_points = np.concatenate((points1, points2), axis=0)

        [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

        H_translation = (np.array([[1, 0, (-x_min)], [0, 1, (-y_min)], [0, 0, 1]])).dot(final_H)

        output_img = self.manual_warp(left_img, H_translation, x_max - x_min, y_max - y_min)
        output_img[(-y_min):rows1 + (-y_min), (-x_min):cols1 + (-x_min)] = right_img

        return output_img, final_H

    def manual_warp(self, src_img, homography, width, height):
        src_rows, src_cols = src_img.shape[:2]
        result_img = np.zeros((height, width, 3), dtype=np.uint8)
        homography_inv = np.linalg.inv(homography)

        for y in range(height):
            for x in range(width):
                original_point = np.array([x, y, 1])
                source_point = homography_inv @ original_point
                source_point /= source_point[2]

                src_x, src_y = int(source_point[0]), int(source_point[1])
                if 0 <= src_x < src_cols and 0 <= src_y < src_rows:
                    result_img[y, x] = src_img[src_y, src_x]

        return result_img

    def get_keypoint(self, left_img, right_img):
        l_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        r_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        key_points1, descriptor1 = sift.detectAndCompute(l_img, None)
        key_points2, descriptor2 = sift.detectAndCompute(r_img, None)
        return key_points1, descriptor1, key_points2, descriptor2

    def match_keypoint(self, key_points1, key_points2, descriptor1, descriptor2):
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptor1, descriptor2, k=2)

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
        u, s, vh = np.linalg.svd(A)
        H = (vh[-1, :].reshape(3, 3))
        H = H / H[2, 2]
        return H

    def ransac(self, good_pts):
        best_inliers = []
        final_H = []
        t = 5
        for i in range(50):
            random_pts = random.sample(good_pts, k=4)
            H = self.homography(random_pts)
            inliers = []
            for pt in good_pts:
                p = np.array([pt[0], pt[1], 1]).reshape(3, 1)
                p_1 = np.array([pt[2], pt[3], 1]).reshape(3, 1)
                Hp = np.dot(H, p)
                Hp = Hp / Hp[2]
                dist = np.linalg.norm(p_1 - Hp)
                if dist < t:
                    inliers.append(pt)
            if len(inliers) > len(best_inliers):
                best_inliers, final_H = inliers, H
        return final_H