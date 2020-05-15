import io
import ntpath
import os

import cv2
import imutils
from PIL import Image
from pylab import *

import utils


# BASE IMAGE ALWAYS IN NPARRAY FORMAT with values 0-255 ! (it's easier to handle)
class CTscanPair:
    # Class attributes

    # Constructor / instances attributes
    #   Pair: tuple with two paths to the preop and postop CT scan images.
    def __init__(self, pair, patternpath):
        self.prePath = pair[1]
        self.postPath = pair[0]
        self.patternPath = patternpath
        self.pattern_arr = []

        head, tail = ntpath.split(self.prePath)
        self.preBasename = tail

        head, tail = ntpath.split(self.postPath)
        self.postBasename = tail

        with open(self.prePath, 'rb') as f:
            self.preop_arr = array(Image.open(io.BytesIO(f.read())).convert('L'))

        with open(self.postPath, 'rb') as g:
            self.postop_arr = array(Image.open(io.BytesIO(g.read())).convert('L'))

        # List of pattern (orientation of inner ear), binary 0-255. Bitwisenot is here to reverse the wrongly made picture
        with open(self.patternPath, 'rb') as p:
            self.pattern = array(Image.open(io.BytesIO(p.read())).convert('L'))
            # self.pattern_arr.append(cv2.bitwise_not(array(Image.open(io.BytesIO(p.read())).convert('L'))))

        self.cochlea_center = self.setSpiralCenter(True, False)

    def getPreImg(self):
        return self.preop_arr

    def getPostImg(self):
        return self.postop_arr

    def getCochleaCenter(self):
        return self.cochlea_center

    # Idea: use template matching (CV2) or normalized cross correlation to find the/a spiral in the picture.
    def setSpiralCenter(self, save_file=False, verbose=False):
        '''
        @:param self:
        @:param save_file: if True, save the scan image with the marked center
        :return: Detected center point of the cochlea
        '''
        # Method used for cv2.matchTemplate -> cv2.TM_CCOEFF
        # Explanations and formulas: https://docs.opencv.org/master/df/dfb/group__imgproc__object.html

        # Noise reduction
        # Strong blur, very efficient!
        img_blur = cv2.blur(self.preop_arr, (65, 65))

        # Apply threshold for grey values
        # Threshold chosen according to image histogram showing peak between 40 and 110 for grey
        # values corresponding to liquid areas.
        img_threshold = np.where(np.logical_or(img_blur > 110, img_blur < 40), 255, 0).astype(np.uint8)

        # Preprocess pattern / no need, it's a simple circle
        # It's an array but only the first value is used (might be useful later)
        template = self.pattern

        # Trying multi scale template matching
        # DISCLAIMER
        # Code adapted from: https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/

        # The best way to detect the spiral is to detect its circle components
        # With one circle, we might encounter the problem that a CT scan image
        # might be bigger than another and the match can be really poor.
        # Using multi scale, avoid this problem and allows to simply give one circle image as input (template)
        # The main part on this function that helped giving good results, was the preprocessing (strong blur)
        # of the preoperative image.

        (tH, tW) = template.shape[:2]

        image = self.preop_arr.copy()
        gray = img_threshold.copy()
        found = None
        visualize = verbose
        # loop over the scales of the image
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            # check to see if the iteration should be visualized
            if visualize:
                # draw a bounding box around the detected region
                clone = np.dstack([edged, edged, edged])
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                              (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                cv2.imshow("Visualize", clone)
                cv2.waitKey(0)
            # if we have found a new maximum correlation value, then update
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
        # unpack the bookkeeping variable and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        # draw a bounding box around the detected result and display the image
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cochlea_center = ((startX + endX) // 2, (startY + endY) // 2)
        cv2.circle(image, cochlea_center, 8, 0, 2)

        if verbose:
            title = "Best match for " + self.preBasename
            utils.show(image, title)
            print("Scale: ", scale)

        # Saving the image
        if save_file:
            path = './GEN_IMG/'
            filename = "center_" + self.preBasename
            os.path.join(path, filename)
            cv2.imwrite(os.path.join(path, filename), image)

        # returns the coordinates of the center
        return cochlea_center
