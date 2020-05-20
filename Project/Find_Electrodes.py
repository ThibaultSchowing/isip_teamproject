import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from imutils import contours
from skimage import measure

# choose file
path = r'C:\Users\stryc\OneDrive\BioSTAT\Image Processing\ISIP 2020 Project Handout\Handout\DATA'
folder = '\ID03'
filename = '\ID03post.png'

# load image
image = cv2.imread(path + folder + filename)

# transform to gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian blurring (find best sigma and kernel size ??)
blurred1 = cv2.GaussianBlur(gray, (21, 21), 0)
blurred2 = cv2.GaussianBlur(gray, (15, 15), sigmaX=0.5)

# thresholding of the images and convert to binary map (find good interval!!)
thresh1 = cv2.threshold(blurred1, 160, 255, cv2.THRESH_BINARY)[1]

print(np.shape(blurred1))

thresh2 = cv2.threshold(blurred2, 170, 255, cv2.THRESH_BINARY)[1]

# potential kernel for morphologyEX, erode and dilate
kernel = np.ones((5, 5), np.uint8)

# opening = cv2.morphologyEx(blurred1, cv2.MORPH_OPEN, None)

# erode reduces noise around dots
thresh1 = cv2.erode(thresh1, None, iterations=1)

# dilate increases dot size after noise reduction > more defined dots (play around with iteration and kernel?)
thresh1 = cv2.dilate(thresh1, None, iterations=1)

# other settings
thresh2 = cv2.erode(thresh2, None, iterations=2)
thresh2 = cv2.dilate(thresh2, None, iterations=4)

# perform a connected component analysis on the thresholded image, then initialize a mask to store only the "large"
# components

# Label connected regions of an integer array.
labels = measure.label(thresh1, background=0)

mask = np.zeros(thresh1.shape, dtype="uint8")

# loop over the unique components
for label in np.unique(labels):

    # if this is the background label, ignore it
    if label == 0:
        continue

    labelMask = np.zeros(thresh1.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)
    print(numPixels)
    # The cluster of pixels should be between 300 and 2500 (needs tweeking)
    if numPixels > 300 and numPixels < 2500:
        mask = cv2.add(mask, labelMask)

# find the contours in the mask, then sort them from left to right
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = contours.sort_contours(cnts)[0]

# loop over the contours
for (i, c) in enumerate(cnts):
    # draw the bright spot on the image

    (x, y, w, h) = cv2.boundingRect(c)
    ((cX, cY), radius) = cv2.minEnclosingCircle(c)
    # cv2.circle(image, (int(cX), int(cY)), int(radius),(0, 0, 255), 3)
    cv2.circle(image, (int(cX), int(cY)), 10, (255, 0, 0), 2)
    cv2.putText(image, "#{}".format(i + 1), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.45, (255, 0, 0), 2)
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)

fig, axs = plt.subplots(2, 3)
axs[0, 0].imshow(image)
axs[0, 0].set_title('image')
axs[0, 1].imshow(blurred1)
axs[0, 1].set_title('blurred1')
axs[0, 2].imshow(blurred2)
axs[0, 2].set_title('blurred2')
axs[1, 0].imshow(labels)
axs[1, 0].set_title('opening')
axs[1, 1].imshow(thresh1)
axs[1, 1].set_title('thresh1')
axs[1, 2].imshow(thresh2)
axs[1, 2].set_title('thresh2')
plt.show()
quit()
