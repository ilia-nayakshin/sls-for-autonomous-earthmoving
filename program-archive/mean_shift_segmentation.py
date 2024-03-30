import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# constants
MAX_DIM_SIZE = 1000 # pixels


# read image and make sure it exists
img = cv.imread('test_img.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"

# make image no larger than MAX_DIM_SIZE in either direction
height, width = img.shape[:2]
print(height, width)
if (width > MAX_DIM_SIZE) or (height > MAX_DIM_SIZE):
    # scale proportionally scale change in other dimension
    if width >= height:
        new_height = int((MAX_DIM_SIZE/width) * height)
        img = cv.resize(img, (MAX_DIM_SIZE, new_height))
        print(MAX_DIM_SIZE, new_height)
    else:
        new_width = int((MAX_DIM_SIZE/height) * width)
        img = cv.resize(img, (new_width, MAX_DIM_SIZE))
        print(new_width, MAX_DIM_SIZE)

# make image greyscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

#COPYPASTE CODE FROM DOCS WEBSITE

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

# apply markers around regions to image
markers = cv.watershed(img,markers)
img[markers == -1] = [0,0,255]

# display result
cv.imshow("bruh", img)
cv.waitKey(0)