import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

def detectFeaturesAndMatch(image1, image2, maxNumOfFeatures=30):
    siftDetector = cv2.SIFT_create()
    keypoints1, descriptors1 = siftDetector.detectAndCompute(image1, None)
    keypoints2, descriptors2 = siftDetector.detectAndCompute(image2, None)
    bruteForceMatcher = cv2.BFMatcher(cv2.NORM_L2)
    rawMatches = bruteForceMatcher.match(descriptors1, descriptors2)
    sortedMatches = sorted(rawMatches, key=lambda x: x.distance)
    featureCorrespondences = []
    for match in sortedMatches:
        featureCorrespondences.append((keypoints1[match.queryIdx].pt, keypoints2[match.trainIdx].pt))
    print(f'Total number of matches: {len(featureCorrespondences)}')
    sourcePoints = np.float32([point_pair[0] for point_pair in featureCorrespondences[:maxNumOfFeatures]]).reshape(-1, 1, 2)
    destinationPoints = np.float32([point_pair[1] for point_pair in featureCorrespondences[:maxNumOfFeatures]]).reshape(-1, 1, 2)
    return np.array(featureCorrespondences[:maxNumOfFeatures]), sourcePoints, destinationPoints
def calculateHomography(point_correspondences):
    matrix_A = []
    for correspondence in point_correspondences:
        src_point, dst_point = correspondence
        src_x, src_y = src_point[0], src_point[1]
        dst_x, dst_y = dst_point[0], dst_point[1]
        matrix_A.append([src_x, src_y, 1, 0, 0, 0, -dst_x * src_x, -dst_x * src_y, -dst_x])
        matrix_A.append([0, 0, 0, src_x, src_y, 1, -dst_y * src_x, -dst_y * src_y, -dst_y])
    matrix_A = np.asarray(matrix_A)
    U, S, Vh = np.linalg.svd(matrix_A)
    homography_elements = Vh[-1, :] / Vh[-1, -1]
    homography_matrix = homography_elements.reshape(3, 3)
    return homography_matrix  
def getBestHomographyRANSAC(correspondences, trials=3000, threshold=2, num_samples=4):
    best_homography = None
    max_inliers = []
    for i in tqdm(range(trials)):
        selected_indices = np.random.choice(len(correspondences), num_samples, replace=False)
        random_sample = correspondences[selected_indices]
        estimated_homography = calculateHomography(random_sample)
        current_inliers = []
        for correspondence in correspondences:
            src_point = np.append(correspondence[0], 1)
            dst_point = np.append(correspondence[1], 1)
            estimated_dst = np.dot(estimated_homography, src_point)
            estimated_dst /= estimated_dst[-1]
            error = np.linalg.norm(dst_point - estimated_dst)
            if error < threshold:
                current_inliers.append(correspondence)
        if len(current_inliers) > len(max_inliers):
            max_inliers = current_inliers
            best_homography = estimated_homography
    print(f"Max inliers = {len(max_inliers)}")
    return best_homography, random_sample
def computeBoundingBoxOfWarpedImage(homography_matrix, img_width, img_height):
    original_corners = np.array([[0, img_width - 1, 0, img_width - 1], 
                                 [0, 0, img_height - 1, img_height - 1], 
                                 [1, 1, 1, 1]])
    transformed_corners = np.dot(homography_matrix, original_corners)
    transformed_corners /= transformed_corners[2, :]
    x_min = np.min(transformed_corners[0])
    x_max = np.max(transformed_corners[0])
    y_min = np.min(transformed_corners[1])
    y_max = np.max(transformed_corners[1])
    return int(x_min), int(x_max), int(y_min), int(y_max)
def warpAndPlaceSourceImage(source_img, homography_matrix, dest_img, use_forward_mapping=False, offset=(2300, 800)):
    height, width, _ = source_img.shape
    homography_inv = np.linalg.inv(homography_matrix)
    if use_forward_mapping:
        coords = np.indices((width, height)).reshape(2, -1)
        homogeneous_coords = np.vstack((coords, np.ones(coords.shape[1])))
        transformed_coords = np.dot(homography_matrix, homogeneous_coords)
        transformed_coords /= transformed_coords[2, :]
        x_output, y_output = transformed_coords.astype(np.int32)[:2, :]
        valid_indices = (x_output >= 0) & (x_output < dest_img.shape[1]) & (y_output >= 0) & (y_output < dest_img.shape[0])
        x_output = x_output[valid_indices] + offset[0]
        y_output = y_output[valid_indices] + offset[1]
        x_input = coords[0][valid_indices]
        y_input = coords[1][valid_indices]
        dest_img[y_output, x_output] = source_img[y_input, x_input]
    else:
        x_min, x_max, y_min, y_max = computeBoundingBoxOfWarpedImage(homography_matrix, width, height)
        coords_x, coords_y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        coords = np.vstack((coords_x.ravel(), coords_y.ravel(), np.ones(coords_x.size)))
        transformed_coords = np.dot(homography_inv, coords)
        transformed_coords /= transformed_coords[2, :]
        x_input = transformed_coords[0].astype(np.int32)
        y_input = transformed_coords[1].astype(np.int32)
        valid = (x_input >= 0) & (x_input < width) & (y_input >= 0) & (y_input < height)
        dest_img[coords_y.ravel()[valid] + offset[1], coords_x.ravel()[valid] + offset[0]] = source_img[y_input[valid], x_input[valid]]
class ImageBlenderWithPyramids():
    def __init__(self, pyramid_depth=6):
        self.pyramid_depth = pyramid_depth
    def getGaussianPyramid(self, image):
        pyramid = [image]
        for _ in range(self.pyramid_depth - 1):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid
    def getLaplacianPyramid(self, image):
        pyramid = []
        for _ in range(self.pyramid_depth - 1):
            next_level_image = cv2.pyrDown(image)
            upsampled_image = cv2.pyrUp(next_level_image, dstsize=(image.shape[1], image.shape[0]))
            laplacian = cv2.subtract(image.astype(float), upsampled_image.astype(float))
            pyramid.append(laplacian)
            image = next_level_image
        pyramid.append(image.astype(float))
        return pyramid
    def getBlendingPyramid(self, laplacian_a, laplacian_b, gaussian_mask_pyramid):
        blended_pyramid = []
        for i, mask in enumerate(gaussian_mask_pyramid):
            triplet_mask = cv2.merge((mask, mask, mask))
            blended_pyramid.append(laplacian_a[i] * triplet_mask + laplacian_b[i] * (1 - triplet_mask))
        return blended_pyramid
    def reconstructFromPyramid(self, laplacian_pyramid):
        reconstructed_image = laplacian_pyramid[-1]
        for laplacian_level in reversed(laplacian_pyramid[:-1]):
            reconstructed_image = cv2.pyrUp(reconstructed_image, dstsize=laplacian_level.shape[:2][::-1]).astype(float) + laplacian_level.astype(float)
        return reconstructed_image
    def generateMaskFromImage(self, image):
        mask = np.all(image != 0, axis=2)
        mask_image = np.zeros(image.shape[:2], dtype=float)
        mask_image[mask] = 1.0
        return mask_image
    def blendImages(self, image1, image2):
        laplacian1 = self.getLaplacianPyramid(image1)
        laplacian2 = self.getLaplacianPyramid(image2)
        mask1 = self.generateMaskFromImage(image1).astype(np.bool_)
        mask2 = self.generateMaskFromImage(image2).astype(np.bool_)
        if mask1.shape != mask2.shape:
            min_shape = np.minimum(mask1.shape, mask2.shape)
            mask1 = mask1[:min_shape[0], :min_shape[1]]
            mask2 = mask2[:min_shape[0], :min_shape[1]]
        overlap_region = mask1 & mask2
        y_coords, x_coords = np.where(overlap_region)
        if len(x_coords) == 0:
            min_x, max_x = image1.shape[1]//2, image1.shape[1]//2
        else:
            min_x, max_x = np.min(x_coords), np.max(x_coords)
        final_mask = np.zeros(image1.shape[:2])
        final_mask[:, :(min_x + max_x)//2] = 1.0
        gaussian_mask_pyramid = self.getGaussianPyramid(final_mask)
        blended_pyramid = self.getBlendingPyramid(laplacian1, laplacian2, gaussian_mask_pyramid)
        blended_image = self.reconstructFromPyramid(blended_pyramid)
        return blended_image, mask1, mask2
class PanaromaStitcher:
    def __init__(self, max_features=30, ratio_thresh=0.75, ransac_thresh=2.0, pyramid_depth=6, warp_size=(600, 400), offset=(2300, 800)):
        self.max_features = max_features
        self.ratio_thresh = ratio_thresh
        self.ransac_thresh = ransac_thresh
        self.pyramid_depth = pyramid_depth
        self.warp_size = warp_size
        self.offset = offset
    def detect_and_match_features(self, img1, img2):
        correspondences, src_pts, dst_pts = detectFeaturesAndMatch(img1,img2, maxNumOfFeatures=self.max_features)
        print(f"Total good matches after filtering: {len(correspondences)}")
        return correspondences
    def compute_homography(self, correspondences):
        if len(correspondences) < 4:
            raise ValueError("At least four correspondences are required to compute homography.")
        H, inliers = getBestHomographyRANSAC(correspondences, trials=3000, threshold=2, num_samples=4)
        if H is None:
            print("Homography could not be computed.")
            return None, None
        print(f"Homography estimated with {len(inliers)} inliers out of {len(correspondences)} matches.")
        return H, inliers
    def warp_images(self, src_img, dest_img, homography):
        warped_img = cv2.warpPerspective(src_img, homography, (dest_img.shape[1], dest_img.shape[0]))
        combined_img = np.maximum(warped_img, dest_img)
        return combined_img
    def blend_panoramas(self, images):
        blender = ImageBlenderWithPyramids(pyramid_depth=self.pyramid_depth)
        panorama = images[0]
        for img in images[1:]:
            panorama, _, _ = blender.blendImages(panorama, img)
        return panorama
    def stitch_and_save_images(self, imagePaths, src_idx, dest_idx, prev_homography, shape, trials=3000, threshold=2, offset=(2300, 800)):
        warped_image = np.zeros((3000, 6000, 3), dtype=np.uint8)
        src_img = cv2.imread(imagePaths[src_idx])
        dest_img = cv2.imread(imagePaths[dest_idx])
        if src_img is None or dest_img is None:
            print(f"Error reading images {imagePaths[src_idx]} or {imagePaths[dest_idx]}")
            return prev_homography
        print(f'Original image size = {src_img.shape}')
        resized_src_img = cv2.resize(src_img, shape)
        resized_dest_img = cv2.resize(dest_img, shape)
        matches, src_pts, dest_pts = detectFeaturesAndMatch(resized_dest_img, resized_src_img, maxNumOfFeatures=self.max_features)
        if len(matches) < 4:
            print(f"Not enough matches between images {src_idx} and {dest_idx}. Skipping.")
            return prev_homography
        best_homography, inliers = getBestHomographyRANSAC(matches, trials=trials, threshold=threshold, num_samples=4)
        if best_homography is None:
            print(f"Homography could not be computed for image pair {src_idx} and {dest_idx}. Skipping.")
            return prev_homography
        new_cumulative_homography = np.dot(prev_homography, best_homography)
        warpAndPlaceSourceImage(resized_dest_img, new_cumulative_homography, dest_img=warped_image, offset=offset)
        scene_number = self.extract_scene_number(imagePaths[0])
        output_dir = f'outputs/scene{scene_number}/custom'
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f'warped_{dest_idx}.png'), warped_image)
        return new_cumulative_homography
    def extract_scene_number(self, image_path):
        scene_number = image_path.split('/')[-2][1]
        return int(scene_number) if scene_number.isdigit() else 1  
    def make_panaroma_for_images_in(self, path):
        imagePaths = sorted(glob.glob(os.path.join(path, '*')))
        num_images = len(imagePaths)
        print(f"Found {num_images} images for stitching.")
        if num_images == 0:
            raise ValueError("No images found in the specified directory.")
        shape = self.warp_size
        trials = 3000  
        threshold = 2
        offset = self.offset
        scene_number = self.extract_scene_number(imagePaths[0])
        if scene_number in [2, 3, 4, 5, 6]:
            prevH = np.eye(3)
            prevH = self.stitch_and_save_images(imagePaths, 2, 1, prevH, shape, trials, threshold, offset)
            prevH = self.stitch_and_save_images(imagePaths, 1, 0, prevH, shape, trials, threshold, offset)
            prevH = np.eye(3)
            prevH = self.stitch_and_save_images(imagePaths, 2, 2, prevH, shape, trials, threshold, offset)
            prevH = self.stitch_and_save_images(imagePaths, 2, 3, prevH, shape, trials, threshold, offset)
            prevH = self.stitch_and_save_images(imagePaths, 3, 4, prevH, shape, trials, threshold, offset)

        else:
            prevH = np.eye(3)
            prevH = self.stitch_and_save_images(imagePaths, 3, 2, prevH, shape, trials, threshold, offset)
            prevH = self.stitch_and_save_images(imagePaths, 2, 1, prevH, shape, trials, threshold, offset)
            prevH = self.stitch_and_save_images(imagePaths, 1, 0, prevH, shape, trials, threshold, offset)
            prevH = np.eye(3)
            prevH = self.stitch_and_save_images(imagePaths, 3, 3, prevH, shape, trials, threshold, offset)
            prevH = self.stitch_and_save_images(imagePaths, 3, 4, prevH, shape, trials, threshold, offset)
            prevH = self.stitch_and_save_images(imagePaths, 4, 5, prevH, shape, trials, threshold, offset)

        print("warping complete")
        print("We are starting with blending")
        output_dir = f'outputs/scene{scene_number}/custom'
        num_warped_images = len(glob.glob(os.path.join(output_dir, 'warped_*.png')))
        images = []
        for idx in range(num_warped_images):
            img_path = os.path.join(output_dir, f'warped_{idx}.png')
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
        finalImg = images[0]
        blender = ImageBlenderWithPyramids(pyramid_depth=self.pyramid_depth)
        for img in images[1:]:
            finalImg, mask1truth, mask2truth = blender.blendImages(finalImg, img)
        cv2.imwrite(os.path.join(output_dir, 'blended_image.png'), finalImg)
        return finalImg, None  
pan = PanaromaStitcher()
a = pan.make_panaroma_for_images_in('/teamspace/studios/this_studio/Images/I2')