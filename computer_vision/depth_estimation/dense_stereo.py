import cv2
import matplotlib.pyplot as plt

leftImg = cv2.imread('tsukuba_l.png', cv2.IMREAD_GRAYSCALE)
rightImg = cv2.imread('tsukuba_r.png', cv2.IMREAD_GRAYSCALE)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=21)
depth = stereo.compute(leftImg, rightImg)
plt.imshow(depth)
plt.axis('off')
plt.show()
