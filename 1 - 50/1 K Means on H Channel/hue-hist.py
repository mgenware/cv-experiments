import numpy as np
import cv2
from matplotlib import pyplot as plt
import colorsys
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

def lab2BGR(img):
    return cv2.cvtColor(img, cv2.COLOR_Lab2BGR)


# load image
img = cv2.imread('../_images/bluebird.jpg')
# convert to HSV
blurred = cv2.GaussianBlur(img, (5, 5), 0)

blocks = np.zeros((img.shape[0], img.shape[1], 1))
blocks[0,0,0] = 1

int noc = 1;
int width = img.shape
