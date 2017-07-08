import numpy as np
import cv2
from matplotlib import pyplot as plt

def nm(img):
   normalizedImg = np.zeros(img.shape)
   return cv2.normalize(img, normalizedImg, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


# load the image in grayscale
img = cv2.imread('../_images/chrome.jpg', 0).astype(np.float32)

sobelH = np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=np.float)
sobelV = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=np.float)

sh = cv2.filter2D(img, -1, sobelH)
sv = cv2.filter2D(img, -1, sobelV)
sm = np.sqrt(sh**2 + sv**2).astype(np.uint8)

centralH = np.array([[0,0,0],[-0.5,0,0.5],[0,0,0]],dtype=np.float)
centralV = np.array([[0,-0.5,0],[0,0,0],[0,0.5,0]],dtype=np.float)

ch = cv2.filter2D(img, -1, centralH)
cv = cv2.filter2D(img, -1, centralV)
cm = np.sqrt(ch**2 + cv**2).astype(np.uint8)

cv2.imshow('Images', np.hstack([nm(img), nm(sh), nm(sv), nm(sm), nm(ch), nm(cv), nm(cm)]))
cv2.waitKey(0)
cv2.destroyAllWindows()
