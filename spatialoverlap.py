import os
import numpy as np
import cv2
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
