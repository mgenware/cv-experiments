import numpy as np
import cv2
from matplotlib import pyplot as plt
import colorsys

# load image
img = cv2.imread('../_images/bluebird.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# kernel
kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.float)
# apply the filter
dst = cv2.filter2D(img, -1, kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Box blur')
plt.xticks([]), plt.yticks([])
plt.show()
