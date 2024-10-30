import numpy as np
import random
import pdb
import glob
import cv2
import os

class PanaromaStitcher:
    def __init__(self):
        # Constructor for the PanoramaStitcher class
        pass

    def make_panaroma_for_images_in(self, path):
        """
        Combines multiple images in the specified directory to create a panorama.
        
        Args:
            directory_path (str): Path to the folder containing images for stitching.
            
        Returns:
            np.array: The final stitched panorama image.
            list: A list of homography matrices used for aligning images.
        """
        image_folder = path
        image_files = sorted(glob.glob(image_folder + os.sep + '*'))
        print(f"Found {len(image_files)} images for panorama creation.")

        if len(image_files) < 2:
            raise ValueError("Stitching requires at least two images.")

        # Start panorama with the first image in the directory
        pano_result = cv2.imread(image_files[0])
        resize_percentage = 50  # Percentage of scaling down each image
        pano_result = cv2.resize(pano_result, None, fx=resize_percentage / 100, fy=resize_percentage / 100)

        homography_matrices = []

        # Process each additional image
        for current_image in image_files[1:]:
            next_image = cv2.imread(current_image)
            if next_image is None:
                print(f"Warning: Unable to load image at {current_image}, skipping.")
                continue
            
            next_image = cv2.resize(next_image, None, fx=resize_percentage / 100, fy=resize_percentage / 100)
            # Stitching the current images
            pano_result, homography_matrix = self.Stitch_2_image_and_matrix_return(pano_result, next_image)
            homography_matrices.append(homography_matrix)

        # Save and output the final stitched image
        cv2.imwrite('stitched_panorama_result.jpg', pano_result)
        print("Panorama image saved as 'stitched_panorama_result.jpg'.")

        return pano_result, homography_matrices 

    def Stitch_2_image_and_matrix_return(self, left_image, right_image):
        """
        Stitches two images together using keypoint matching and homography estimation.
        
        Args:
            left_image (np.array): The base image.
            right_image (np.array): The image to align and stitch with the base image.
            
        Returns:
            np.array: The resulting stitched image.
            np.array: The homography matrix used for alignment.
        """
        # Detect and describe keypoints in both images
        kp1, desc1, kp2, desc2 = self.obtain_the_key_points(left_image, right_image)

        # Match keypoints using nearest neighbors
        keypoint_matches = self.align_and_match_feature_points(kp1, kp2, desc1, desc2)

        # Estimate the homography with RANSAC to remove outliers
        optimal_H = self.implementated_Ransac(keypoint_matches)

        # Determine image dimensions
        img_height1, img_width1 = right_image.shape[:2]
        img_height2, img_width2 = left_image.shape[:2]

        # Define the corners for each image to find the combined boundaries
        corners_image1 = np.float32([[0, 0], [0, img_height1], [img_width1, img_height1], [img_width1, 0]]).reshape(-1, 1, 2)
        corners_image2 = np.float32([[0, 0], [0, img_height2], [img_width2, img_height2], [img_width2, 0]]).reshape(-1, 1, 2)

        # Apply the homography to the corners of the second image
        transformed_corners = cv2.perspectiveTransform(corners_image2, optimal_H)
        combined_corners = np.concatenate((corners_image1, transformed_corners), axis=0)

        # Calculate the bounding box of the final stitched image
        [x_min, y_min] = np.int32(combined_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(combined_corners.max(axis=0).ravel() + 0.5)

        # Adjust transformation matrix to include translations
        translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]).dot(optimal_H)

        # Warp the first image to fit within the bounding box
        final_output = self.apply_warp(left_image, translation_matrix, x_max - x_min, y_max - y_min)

        # Place the second image onto the panorama
        final_output[(-y_min):img_height1 + (-y_min), (-x_min):img_width1 + (-x_min)] = right_image
        return final_output, optimal_H

    def apply_warp(self, source_img, homography_matrix, output_width, output_height):
        """
        Warps the source image based on the homography matrix.

        Args:
            source_img (np.array): The image to be warped.
            homography_matrix (np.array): Homography matrix for transformation.
            output_width (int): Width of the final panorama.
            output_height (int): Height of the final panorama.

        Returns:
            np.array: The warped image with a black background.
        """
        source_height, source_width = source_img.shape[:2]
        warped_img = np.zeros((output_height, output_width, 3), dtype=np.uint8)

        # Calculate the inverse of the homography matrix
        inverse_H = np.linalg.inv(homography_matrix)

        # Apply inverse transform for each pixel location in the output
        for y in range(output_height):
            for x in range(output_width):
                destination_pt = np.array([x, y, 1])
                original_pt = inverse_H @ destination_pt
                original_pt /= original_pt[2]

                src_x, src_y = int(original_pt[0]), int(original_pt[1])

                # Only copy pixel if within source image boundaries
                if 0 <= src_x < source_width and 0 <= src_y < source_height:
                    warped_img[y, x] = source_img[src_y, src_x]

        return warped_img

    def obtain_the_key_points(self, left_image, right_image):
        """
        Extracts keypoints and descriptors from both images using SIFT.
        
        Args:
            left_image (np.array): First image for keypoint detection.
            right_image (np.array): Second image for keypoint detection.
        
        Returns:
            tuple: Keypoints and descriptors from both images.
        """
        # Convert images to grayscale
        grayscale_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        grayscale_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        # Initialize SIFT detector
        sift_detector = cv2.SIFT_create()
        kp1, desc1 = sift_detector.detectAndCompute(grayscale_left, None)
        kp2, desc2 = sift_detector.detectAndCompute(grayscale_right, None)

        return kp1, desc1, kp2, desc2

    def align_and_match_feature_points(self, kp1, kp2, desc1, desc2):
        """
        Matches keypoints between two images using FLANN-based matching.
        
        Args:
            kp1 (list): Keypoints from the first image.
            kp2 (list): Keypoints from the second image.
            desc1 (np.array): Descriptors of keypoints from the first image.
            desc2 (np.array): Descriptors of keypoints from the second image.
        
        Returns:
            list: A list of matched keypoint coordinates.
        """
        # Configure the FLANN matcher
        flann_index_params = dict(algorithm=1, trees=5)
        flann_search_params = dict(checks=50)
        flann_matcher = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)

        # Match descriptors using k-nearest neighbors
        knn_matches = flann_matcher.knnMatch(desc1, desc2, k=2)
        filtered_matches = []

        for match_1, match_2 in knn_matches:
            if match_1.distance < 0.75 * match_2.distance:  # Ratio test for filtering
                left_coords = kp1[match_1.queryIdx].pt
                right_coords = kp2[match_1.trainIdx].pt
                filtered_matches.append([left_coords[0], left_coords[1], right_coords[0], right_coords[1]])

        return filtered_matches

    def calculate_homography(self, matched_points):
        """
        Calculates the homography matrix from a set of matched points.

        Args:
            matched_points (list): Matched points between images.
        
        Returns:
            np.array: The 3x3 homography matrix.
        """
        equation_matrix = []
        for match in matched_points:
            x, y = match[0], match[1]
            X, Y = match[2], match[3]
            equation_matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
            equation_matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])

        equation_matrix = np.array(equation_matrix)
        _, _, v_transpose = np.linalg.svd(equation_matrix)
        homography_matrix = (v_transpose[-1, :].reshape(3, 3))
        homography_matrix = homography_matrix / homography_matrix[2, 2]  # Normalize
        return homography_matrix

    def implementated_Ransac(self, matched_points):
        """
        Uses RANSAC to compute the best homography matrix and filter outliers.

        Args:
            matched_points (list): List of matched keypoint coordinates.

        Returns:
            np.array: The homography matrix with the most inliers.
        """
        max_inliers = []
        best_homography = []
        threshold_dist = 5
        num_iterations = 50  # Can adjust based on preference
        for _ in range(num_iterations):
            sample_points = random.sample(matched_points, k=4)  # Random sample
            H = self.calculate_homography(sample_points)
            current_inliers = []

            for pt in matched_points:
                origin_pt = np.array([pt[0], pt[1], 1]).reshape(3, 1)
                target_pt = np.array([pt[2], pt[3], 1]).reshape(3, 1)
                transformed_pt = np.dot(H, origin_pt)
                transformed_pt /= transformed_pt[2]
                point_distance = np.linalg.norm(target_pt - transformed_pt)

                if point_distance < threshold_dist:
                    current_inliers.append(pt)

            # Retain the homography with maximum inliers
            if len(current_inliers) > len(max_inliers):
                max_inliers, best_homography = current_inliers, H

        return best_homography
