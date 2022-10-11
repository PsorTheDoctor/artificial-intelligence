import cv2

img = cv2.imread('licencePlate1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

adaptiveMean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 199, 5)
adaptiveGaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 199, 5)

_, otsu = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('Adapt Mean', adaptiveMean)
cv2.imshow('Adapt Gaussian', adaptiveGaussian)
cv2.imshow('Otsu', otsu)
cv2.waitKey(0)
