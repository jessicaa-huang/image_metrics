import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_mutual_information(image1, image2, bins=20):
    """Compute mutual information between two images."""
    hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=bins)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # Marginal for x over y
    py = np.sum(pxy, axis=0)  # Marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def select_top_k_images(folder_path, reference_image_path, k):
    """Select top k images with the highest mutual information."""
    # Load reference image
    ref_img = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        print(reference_image_path)
        raise ValueError("Reference image could not be loaded.")
    
    # List to store mutual information scores
    mi_scores = []
    image_paths = []
    
    # Process all images in the folder
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # Skip if the image could not be loaded
        
        # Resize to match reference image if needed
        if img.shape != ref_img.shape:
            img = cv2.resize(img, (ref_img.shape[1], ref_img.shape[0]))
        
        # Compute mutual information
        mi = compute_mutual_information(ref_img, img)
        mi_scores.append((mi, img_name))
        image_paths.append(img_path)
    
    # Sort by mutual information in descending order
    mi_scores.sort(reverse=True, key=lambda x: x[0])
    
    # Select top k images
    top_k = mi_scores[:k]
    return top_k, image_paths

# Example usage
# folder = "ground"
# k = 25
testPath = "University-Release/train/drone/0839"
refPath = "University-Release/train/drone/0839/image-01.jpeg"

def plot_MI(path=testPath, refImg=refPath, k=25):
    # Process all images in the folder
    # img_count = 0
    # for _ in os.listdir(path):
    #     img_count += 1
    k = k + 1

    top_k_images, image_paths = select_top_k_images(path, refImg, k)

    # Extract the MI scores
    mi_values = [mi for mi, _ in top_k_images][1:]
    image_paths = [pt for _, pt in top_k_images][1:]

    print(mi_values)
    print(image_paths)
    
    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(mi_values, bins=10, color='blue', alpha=0.7, edgecolor='black')
    plt.title('Histogram of Mutual Information (MI) Scores')
    plt.xlabel('Mutual Information')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    # plt.yticks(range(0, k + 1))  # Tick values from 0 to k
    plt.show()


# result = []
# img_topK = {}

# for i in range(2, 10, 2):
#     reference_image = f"ground/ground{i}.png"
#     ref = f"ground{i}.png"
    
#     # Check if the reference image exists
#     if not os.path.exists(reference_image):
#         continue
    
#     top_k_images, image_paths = select_top_k_images(folder, reference_image, k)
#     img_topK[ref] = top_k_images
    
#     totalSum, count = 0, 0
#     for mi, img_name in top_k_images:
#         totalSum += mi
#         count += 1
    
#     # Compute and print the average MI for the current reference
#     if count > 0:
#         average = totalSum / count
#         print(f"Average with {reference_image} as the reference: {average}")
#         result.append([average, ref])
# result.sort(reverse=True)
# print(result)
# # Print only the names of the top images for the reference with the highest average MI
# topAverage, ref = result[0]
# print()
# print(f"The best reference image to use is: {ref}. These are the top {k} images to use.")
# top_images = [img_name for _, img_name in img_topK[ref]]  # Extract only image names
# print(top_images)
