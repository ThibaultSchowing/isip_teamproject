import io
import ntpath
import os

import cv2
import imutils
from PIL import Image
from pylab import *

import config
import utils


# BASE IMAGE ALWAYS IN NPARRAY FORMAT with values 0-255 ! (it's easier to handle)

# This class contains all information related to the pair of images received (pre-surgical and post-surgical)
class CTscanPair:
    # Class attributes

    # Constructor / instances attributes
    def __init__(self, pair, patternpath):
        """

        :param pair: Tuple with two paths to the preop and postop CT scan images.
        :param patternpath: Path to the pattern images to detect the cochlea (circle)
        """
        ###########################################
        # Basic features initialization
        ###########################################
        start = time.time()
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

        # Base image preop but colored / is set during cochlea center calculation bellow
        self.preImgRGB = None

        ###########################################
        # Complex analysis on init
        ###########################################

        # Define a radius of reasonable location of the cochlea (see hand out: electrode might not be in)

        # Has been attempted with Hough Transform but template matching is more efficient and adaptive.
        self.cochlea_center = self.setCochleaCenterTemplateMatching()

        # Mask that covers the cochlea area. It is expanded a bit to be used in the electrodes detection.
        self.cochlea_area = self.setCochleaAreaImage()
        # self.electrodes_list = self.setElectrodesCoordinates()

        self.orientation = self.setCochleaOrientation()

        end = time.time()
        t = end - start
        message = " Done for pair " + self.preBasename[:-7] + " in " + str(round(t, 3)) + " seconds. "
        print(message)

    def getPreImg(self):
        """

        :return: Base image of the pre-surgery CT scan (without electrodes)
        """
        return self.preop_arr

    def getPostImg(self):
        """

        :return: Base image of the post-surgery CT scan (with electrodes)
        """
        return self.postop_arr

    def getCochleaCenter(self):
        """

        :return: Coordinates (x,y) of the cochleal center
        """
        return self.cochlea_center

    # HOUGH IS NOT USED
    def setCochleaCenterHoughTransform(self):
        """
        Circle detection with Hough Transform
        This method has been tried but getting circles was not successful enough.
        This function is here for academic purpose but does not serve at all to the project.
        :return: 8 (because why not)
        """
        # Noise reduction
        # Strong blur, very efficient (65, 65)
        img_blur = cv2.blur(self.preop_arr, (config.preprocessing_1["blur"],
                                             config.preprocessing_1["blur"]))

        # Apply threshold for grey values
        # Threshold chosen according to image histogram showing peak between 40 and 110 for grey
        # values corresponding to liquid areas. (between 40 and 110)
        hough_gray = np.where(np.logical_or(img_blur > config.preprocessing_1["thr_up_gray"],
                                            img_blur < config.preprocessing_1["thr_low_gray"]),
                              255,
                              0).astype(np.uint8)

        # Canny edge detection
        hough_gray = cv2.Canny(hough_gray,
                               config.hough_circles_cochlea_center["canny_thr1"],
                               config.hough_circles_cochlea_center["canny_thr2"])

        # copy of original image
        output = self.preop_arr.copy()
        image = output.copy()

        # detect circles in the image
        circles = cv2.HoughCircles(hough_gray,
                                   config.hough_circles_cochlea_center["method"],
                                   config.hough_circles_cochlea_center["accumulator_value"],
                                   config.hough_circles_cochlea_center["min_dist"])

        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # show the output image
            cv2.imshow("output hough transform", np.hstack([image, output]))
            cv2.waitKey(0)
        else:
            if config.hough_circles_cochlea_center["verbose"]:
                print("No circle deteted")

        return 8

    def setCochleaCenterTemplateMatching(self,
                                         save_file=config.pattern_matching_cochlea_center["save_file"],
                                         verbose=config.pattern_matching_cochlea_center["verbose"],
                                         show_plot=config.pattern_matching_cochlea_center["show_plot"]):
        """

        :param save_file:
        :param verbose:
        :param show_plot:
        :return: Detected center point of the cochlea
        """
        # Method used for cv2.matchTemplate -> cv2.TM_CCOEFF
        # Explanations and formulas: https://docs.opencv.org/master/df/dfb/group__imgproc__object.html

        # Noise reduction
        # Strong blur, very efficient (65, 65)
        img_blur = cv2.blur(self.preop_arr, (config.pattern_matching_cochlea_center["blur"],
                                             config.pattern_matching_cochlea_center["blur"]))

        # Apply threshold for grey values
        # Threshold chosen according to image histogram showing peak between 40 and 110 for grey
        # values corresponding to liquid areas. (between 40 and 110)
        img_threshold = np.where(np.logical_or(img_blur > config.pattern_matching_cochlea_center["thr_up_gray"],
                                               img_blur < config.pattern_matching_cochlea_center["thr_low_gray"]),
                                 255,
                                 0).astype(np.uint8)

        # Preprocess pattern / no need, it's a simple circle
        template = self.pattern

        # Trying multi scale template matching
        # DISCLAIMER
        # Code adapted from: https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/

        # Important documentation:
        # https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny#canny
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html

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
        for scale in config.pattern_matching_cochlea_center["image_scaling"]:
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
            # !!!! Parameters from config.py !!!!!
            edged = cv2.Canny(resized,
                              config.pattern_matching_cochlea_center["canny_thr1"],
                              config.pattern_matching_cochlea_center["canny_thr2"])

            result = cv2.matchTemplate(edged,
                                       template,
                                       config.pattern_matching_cochlea_center["pat_match_method"])

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

            filenamep = "preprocessed_" + self.preBasename
            cv2.imwrite(os.path.join(path2, filenamep), img_threshold)

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
        radius_blue = config.pattern_matching_cochlea_center["mask_radius"]
        mask = utils.create_circular_mask(self.preImgRGB.shape[0], self.preImgRGB.shape[1], cochlea_center, radius_blue)

        # Mask2: Region marked as "liquid", img_threshold had to be reversed
        mask2 = -1 * (img_threshold - 255)

        # Logical and on the two masks to obtain the region shown in blue in the picture that
        # corresponds to the cochlea. (liquid + matching radius
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

    # TODO - this is just bullshit
    def setElectrodesCoordinates(self):
        #
        img = self.postop_arr.copy()

        # increase contrast
        normalizedImg = np.zeros(img.shape)
        normalizedImg = cv2.normalize(img, normalizedImg, 0, 255, cv2.NORM_MINMAX)
        img = normalizedImg

        # global thresholding
        ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # Otsu's thresholding
        ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img, (5, 5), 6)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # plot all the images and their histograms
        images = [img, 0, th1,
                  img, 0, th2,
                  blur, 0, th3]
        titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
                  'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
                  'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
        for i in range(3):
            plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
            plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
            plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
            plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
            plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
            plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
        plt.show()

        # threshold
        # utils.get_image_info(img)
        # ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)

        # findcontours
        # img = cv2.findContours(img, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2]

        # create a CLAHE object (Arguments are optional).
        # clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(4, 4))
        # img = clahe.apply(img)

        # Normalizer

        # increase contrast
        # normalizedImg = np.zeros(img.shape)
        # normalizedImg = cv2.normalize(img, normalizedImg, 0, 255, cv2.NORM_MINMAX)
        # img = normalizedImg

        # Equalizers
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html

        # img = cv2.equalizeHist(img)

        # Bad idea, some surgeries are really fucked up
        # radius = 450
        # mask = utils.create_circular_mask(self.postop_arr.shape[0],self.postop_arr.shape[1],self.cochlea_center, radius)
        # img = np.where(mask == 0, 0, img)

        # create a CLAHE object (Arguments are optional).
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # img = clahe.apply(img)

        # sharpen the image
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # img = cv2.filter2D(img, -1, kernel)

        # img = utils.normalize_2dimage_grayscale(img)

        # img = np.where(img > 245,255,0).astype(np.uint8)

        # img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]
        # img = cv2.erode(img, None, iterations=2)
        # img = cv2.dilate(img, None, iterations=2)

        utils.show(img)

        # imgd = np.hstack([img, self.postop_arr])
        # utils.show(imgd)

        # set the coordinate list instead of returning stuff
        return 8

    # TODO - make this mean something
    def setElectrodesOrder(self):

        return 8

    # TODO - separate bright spots finder
    # https: // stackoverflow.com / questions / 51846933 / finding - bright - spots - in -a - image - using - opencv

    # TODO - find orientation of the cochlea
    def setCochleaOrientation(self):

        pass

    # todo - set cochlea area image
    def setCochleaAreaImage(self):
        center = self.cochlea_center
        # TODO - same as in center detection -> need to be compacted if time allows it (I doubt it)
        # Noise reduction
        # Strong blur, very efficient (65, 65)
        img_blur = cv2.blur(self.preop_arr, (config.cochlea_area["blur"],
                                             config.cochlea_area["blur"]))

        # Apply threshold for grey values
        # Threshold chosen according to image histogram showing peak between 40 and 110 for grey
        # values corresponding to liquid areas. (between 40 and 110)
        img_threshold = np.where(np.logical_or(img_blur > config.cochlea_area["thr_up_gray"],
                                               img_blur < config.cochlea_area["thr_low_gray"]),
                                 0,
                                 255).astype(np.uint8)

        # Mask: circle around the center BOOLEAN (True/False)
        radius_blue = config.cochlea_area["mask_radius"]
        mask = utils.create_circular_mask(self.preImgRGB.shape[0], self.preImgRGB.shape[1], center, radius_blue)

        # Logical and on the two masks to obtain the region shown in blue in the picture that
        # corresponds to the cochlea. (liquid + matching radius
        mask3 = np.logical_and(mask, img_threshold)

        # before = mask3.copy()

        # As said in the handout, the electrodes might not be exactely in the cochlea area
        # So we thicken it to have extend the borders of the mask.

        kernel = np.ones((19, 19), np.uint8)
        mask3 = cv2.dilate(np.float32(mask3), kernel, iterations=config.cochlea_area["iterations"])
        # Convert back to boolean
        # Type: <class 'numpy.ndarray'>
        # Shape: (746, 1129)
        # dType: bool

        mask3 = mask3.astype(bool)

        # after = mask3.copy()
        # utils.show(np.hstack([before, after]), "Cochlea area mask before and after thickening")

        # Save to report
        # Usage:

        # Saving the image
        if config.cochlea_area["save_file"]:
            image = np.where(mask3, self.postop_arr, 0)
            filename = "area_mask_" + self.preBasename
            cv2.imwrite(os.path.join(config.general["save_imgs"], filename), image)

        return mask3
