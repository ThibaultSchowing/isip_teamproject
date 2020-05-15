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
    def __init__(self, pair, patternpath):
        '''

        :param pair: Tuple with two paths to the preop and postop CT scan images.
        :param patternpath: Path to the pattern images to detect the cochlea (circle)
        '''
        ###########################################
        # Basic features initialization
        ###########################################
        self.prePath = pair[1]
        self.postPath = pair[0]
        self.patternPath = patternpath
        self.pattern_arr = []

        # Preop picture filename
        head, tail = ntpath.split(self.prePath)
        self.preBasename = tail
        # Postop picture filename
        head, tail = ntpath.split(self.postPath)
        self.postBasename = tail

        # Open the files
        # https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.convert

        # Pre-surgery picture
        with open(self.prePath, 'rb') as f:
            self.preop_arr = array(Image.open(io.BytesIO(f.read())).convert('L'))

        # Post-surgery picture
        with open(self.postPath, 'rb') as g:
            self.postop_arr = array(Image.open(io.BytesIO(g.read())).convert('L'))

        # Pattern (circle)
        with open(self.patternPath, 'rb') as p:
            self.pattern = array(Image.open(io.BytesIO(p.read())).convert('L'))

        # Base image preop but colored / is set during cochlea center calculation
        self.preImgRGB = None

        ###########################################
        # Complex analysis on init
        ###########################################

        self.cochlea_center = self.setCochleaCenter(True, False, False)

    def getPreImg(self):
        '''

        :return: Base image of the pre-surgery CT scan (without electrodes)
        '''
        return self.preop_arr

    def getPostImg(self):
        '''

        :return: Base image of the post-surgery CT scan (with electrodes)
        '''
        return self.postop_arr

    def getCochleaCenter(self):
        '''

        :return: Coordinates (x,y) of the cochleal center
        '''
        return self.cochlea_center

    # Called in constructor
    def setCochleaCenter(self, save_file=False, verbose=False, show_plot=False):
        '''
        @:param self:
        @:param save_file: if True, save the scan image with the marked center + with colored area
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

        # A trial with Hough transform gave poor results. It was hard to score the different circles
        # and to tune the functions. With a bit more or different pre-processing steps, Hough transform
        # might also give satisfying results but the one we get here are satisfactory enough.

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
        cv2.rectangle(image, (startX + 100, startY + 100), (endX - 100, endY - 100), (0, 0, 255), 2)
        cochlea_center = ((startX + endX) // 2, (startY + endY) // 2)
        cv2.circle(image, cochlea_center, 8, 0, 2)

        if verbose:
            title = "Best match for " + self.preBasename
            utils.show(image, title)

        # Saving the image
        if save_file:
            path2 = './GEN_IMG/'
            filename = "center_" + self.preBasename
            cv2.imwrite(os.path.join(path2, filename), image)

        #################################################################
        # Create picture with colored center and cochlea
        #################################################################
        # Copy of the original picture that we'll modify here
        self.preImgRGB = cv2.cvtColor(self.preop_arr, cv2.COLOR_GRAY2RGB)
        cv2.circle(self.preImgRGB, cochlea_center, 8, [0, 100, 150], 2)

        # Create masks to extract only the cochleal area
        mask = np.zeros((self.preImgRGB.shape[0], self.preImgRGB.shape[1]))

        # KEEP
        # If the wanted shape is rectangle
        # Fairly it's useless but it's a pain in the ass to write so I keep it there
        rectangle = False
        if rectangle:
            pad_rectangle = 50
            mask[startY + pad_rectangle:endY - pad_rectangle, startX + pad_rectangle:endX - pad_rectangle] = 1

        # Mask: circle around the center
        radius_blue = 250
        mask = utils.create_circular_mask(self.preImgRGB.shape[0], self.preImgRGB.shape[1], cochlea_center, radius_blue)

        # Mask2: Region marked as "liquid", img_threshold had to be reversed
        mask2 = -1 * (img_threshold - 255)

        # Logical and on the two masks to obtain the region shown in blue in the picture that
        # corresponds to the cochlea.
        mask3 = np.logical_and(mask, mask2)
        mask3 = cv2.cvtColor(np.float32(mask3), cv2.COLOR_GRAY2RGB)
        background = np.int32(self.preImgRGB.copy())
        overlay = np.where(mask3, [0, 100, 150], [0, 0, 0])

        # In overlay, it would be great to remove the small blue parts outside the cochlea
        # An easy way to do this would be to use an algorithm that detect islands of connected 1's
        # -> not enough time to do this because it's the exams soon sorry.

        # It is important to check for the image type (float32, int8, etc)
        # print("bck type: ", background.dtype)
        # print("ovl type: ", overlay.dtype)

        # Superpose the blue mask and the picture

        img_colored = cv2.addWeighted(background, 1, overlay, 0.2, 0)
        self.preImgRGB = img_colored.copy()

        # Useful for debug
        # utils.get_image_info(img_colored)

        img_colored = img_colored.astype('uint8')

        # Due to the RGB vs BGR thing, the picture here is displayed blue
        # but saved yellow (it is in RGB for openCV but BGR for every other program in the world)
        if show_plot:
            utils.show(img_colored, "Colored cochlea and center")

        # Saving the image
        if save_file:
            # Convert the image before saving
            # Will appear blue when opened but would appear yellow if ploted
            img_colored = cv2.cvtColor(img_colored, cv2.COLOR_RGB2BGR)

            path2 = './GEN_IMG/'
            filename = "center_colored_" + self.preBasename
            cv2.imwrite(os.path.join(path2, filename), img_colored)

        # Has set the variable self.preImgRGB to a colored preop image
        # returns the coordinates of the center
        return cochlea_center
