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

# plot
hPlot = data.reshape((-1, 1))
_, bins, patches = plt.hist(hPlot, 180, [0, 180])
for bin, patch in zip(bins, patches):
    plt.setp(patch, 'facecolor', colorsys.hsv_to_rgb(bin / 180, 1.0, 1.0))
plt.show()
