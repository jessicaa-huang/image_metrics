import os
import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt

def compute_spatial_overlap(image1, image2):
    if image1 is None or image2 is None:
        raise ValueError("One or both input images are None.")
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    overlap = len(matches) / min(len(keypoints1), len(keypoints2))
    return 1 - overlap 

def spatial_overlap(folder_path, ref_image):
    result = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        result.append(compute_spatial_overlap(ref_image, img))
    return result

def filter_images_by_spatial_overlap(satellite_image_path, drone_images_folder, output_folder, threshold=0.65):
    # Load the satellite image
    satellite_image = cv2.imread(satellite_image_path, cv2.IMREAD_GRAYSCALE)
    if satellite_image is None:
        raise ValueError(f"Satellite image at {satellite_image_path} could not be loaded.")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    countUsed = 0
    totalCount = 0

    for img_name in os.listdir(drone_images_folder):
        img_path = os.path.join(drone_images_folder, img_name)
        drone_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if drone_image is None:
            print(f"Skipping {img_name}: Unable to load image.")
            continue

        # Calculate spatial overlap
        overlap = compute_spatial_overlap(satellite_image, drone_image)
        
        if overlap > threshold:
            # Copy image to the output folder if it passes the threshold
            output_path = os.path.join(output_folder, img_name)
            shutil.copy(img_path, output_path)
            countUsed += 1
            print(f"Copied {img_name} to {output_folder}: Overlap = {overlap:.2f}")
        else:
            print(f"Skipped {img_name}: Overlap = {overlap:.2f}")
        totalCount += 1

    print(f"Used {countUsed} images out of {totalCount}")
def compute_feature_density(image): 
    orb = cv2.ORB_create() 
    keypoints = orb.detect(image, None) 
    return len(keypoints)

def compute_image_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    entropy = -np.sum(hist * np.log2(hist + 1e-8))
    return entropy

# def compute_color_histogram_diversity(image1, image2):
#     hist1 = cv2.calcHist([image1], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
#     hist2 = cv2.calcHist([image2], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
#     hist1 = cv2.normalize(hist1, hist1).flatten()
#     hist2 = cv2.normalize(hist2, hist2).flatten()
#     return entropy(hist1, hist2)
