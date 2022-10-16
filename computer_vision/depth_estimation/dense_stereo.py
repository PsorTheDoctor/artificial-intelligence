import cv2
import matplotlib.pyplot as plt

leftImg = cv2.imread('tsukuba_l.png', cv2.IMREAD_GRAYSCALE)
rightImg = cv2.imread('tsukuba_r.png', cv2.IMREAD_GRAYSCALE)

# Brute Matching
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=21)
depth = stereo.compute(leftImg, rightImg)
plt.figure()
plt.title('Brute Matching')
plt.imshow(depth)
plt.axis('off')

# Semi-Global Brute Matching
stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=21)
depth = stereo.compute(leftImg, rightImg)
plt.figure()
plt.title('Semi-Global Brute Matching')
plt.imshow(depth)
plt.axis('off')
plt.show()
