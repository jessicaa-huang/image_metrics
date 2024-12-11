import cv2
import numpy as np

def compute_kl_divergence(img1, img2):
   # Convert images to grayscale
   img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
   img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

   # Calculate histograms
   hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
   hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])

   # Normalize histograms
   hist1 = hist1 / hist1.sum()
   hist2 = hist2 / hist2.sum()

   # Calculate KL Divergence
   kl_divergence = np.sum(np.where(hist1 != 0, hist1 * np.log(hist1 / hist2), 0))

   return kl_divergence

# # Load images
# img1 = cv2.imread('image1.jpg')
# img2 = cv2.imread('image2.jpg')

# kl_divergence = compute_kl_divergence(img1, img2)

# print("KL Divergence:", kl_divergence)