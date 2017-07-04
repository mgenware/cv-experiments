import numpy as np
import cv2
from matplotlib import pyplot as plt
import colorsys

# load image
img = cv2.imread('bluebird.jpg')
# convert to HSV
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# extract H channel
data = np.float32(img.reshape((-1, 3)))
data = data[:,0]

# define criteria 
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
# apply k means
_, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# plot
centers = centers.flatten()
hPlot = data.reshape((-1, 1))
for i, center in enumerate(centers):
  plt.hist(hPlot[labels == i], 180, [0, 180], color=colorsys.hsv_to_rgb(center / 180, 1.0, 1.0))

for center in centers:
  plt.axvline(x=center, color='r', linestyle='dashed', linewidth=1)

plt.show()
