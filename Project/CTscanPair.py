import io
import ntpath

import cv2
import imutils
from PIL import Image
from pylab import *
from skimage import measure

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
        self.name = self.preBasename[:-7]

        print("\n\n-------------------------------------")
        print("Scan pair identification number: ", self.name)

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
        # Complex analysis and definitions on init
        ###########################################


        # Has been attempted with Hough Transform but template matching is more efficient and adaptive.
        self.cochlea_center = self.setCochleaCenterTemplateMatching()

        # Define a radius of reasonable location of the cochlea (see hand out: electrode might not be in)
        # Mask that covers the cochlea area. It is expanded a bit to be used in the electrodes detection.
        self.cochlea_area = self.setCochleaAreaImage()

        # Get coordinates of the electrodes. A list of x and y coordinates (tuples) pinpointing to the center of the
        # electrode.
        self.electrodes_list = self.setElectrodesCoordinates()

        # Determine if the found spots are relevant --> 12 electrodes should be found! Not too far away
        self.relevant_electodes = self.setElectrodesSorted()

        # Determine the orientation of the cochlea
        self.orientation = self.setCochleaOrientation()

        # Define the order of the electrodes
        self.electrode_order = self.setElectrodesOrder()

        # Calculate the angular insertion depth --> The main output!
        # [(electrode nr., x-cord., y-cord., angular insertion depth),...]
        self.angular_insertion_depth = self.setAngularInsertionDepth()

        # If wanted, found electrodes can be visualized
        if config.electrodes_enumeration["Show found electrodes on image?"]:
            self.enumerateElectrodes()

        end = time.time()
        t = end - start
        message = "Done for pair " + self.preBasename[:-7] + " in " + str(round(t, 3)) + " seconds. " + "\n-------------------------------------"
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

    def getElectrodesCoordinates(self):
        """

        :return: Cordinates of (x,y) of detected electrodes
        """
        return self.electrodes_list

    def getCochleaOrientation(self):
        """

        :return: Returns the orientation of the cochlea in the provided image.
        """
        return self.orientation

    def getElectrodesSorted(self):
        """

        :return:
        """
        return self.relevant_electodes

    def getAngularInsertionDepth(self):
        return self.angular_insertion_depth

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
            title_ = "Best match for " + self.preBasename
            utils.show(image, title_)

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

    def new_version_setElectrodesCoordinates(self):
        """
        :return: Same than previous version but performs worst at some point. Uses cv2.SimpleBlobDetector.
        """

        # get post op image and invert intensity
        img = self.postop_arr.copy()
        img = cv2.bitwise_not(img)

        ###############################################
        # Setup SimpleBlobDetector parameters.
        parameters = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        parameters.minThreshold = 0
        parameters.maxThreshold = 255

        # Filter by Area.
        parameters.filterByArea = True
        parameters.minArea = 10

        # Filter by Circularity
        parameters.filterByCircularity = True
        parameters.minCircularity = 0.1

        # Filter by Convexity
        parameters.filterByConvexity = False
        parameters.minConvexity = 0.87

        # Filter by Inertia
        parameters.filterByInertia = False
        parameters.minInertiaRatio = 0.01
        ###############################################################

        # create mask of the cochlea for normalization. The cochlea is the region that is of interest to us, so we use
        # it to normalize the images. This region contains (most of the) electrodes as well as the background noise that
        # surrounds them.
        cochlea_mask = self.cochlea_area
        normRegion = np.where(cochlea_mask == 0, cochlea_mask, img)

        # Calculate mean and STD of normRegion for clipping and normalization
        mean, SD = cv2.meanStdDev(normRegion)

        # we clip the pixel values outside of the interval [mean-SD, mean+SD]
        clipped = np.clip(img, mean - 2 * SD, mean + 2 * SD).astype(np.uint8)

        # Normalize the image
        mean_cl, SD_cl = cv2.meanStdDev(clipped)
        img_norm = cv2.normalize(clipped, clipped, mean_cl, 255, norm_type=cv2.NORM_MINMAX)

        # Gaussian blurring to reduce high frequency noise
        blurred_img = cv2.GaussianBlur(img_norm, (41, 41), 0)

        # Extract statistics of the normalized image
        min_img, max_img = np.amin(blurred_img), np.amax(blurred_img)
        mean_img, SD_img = cv2.meanStdDev(blurred_img)

        # thresholding of the images and conversion to a binary image
        # 3SD away from max value of image (= 255) seems reasonable
        thresh_img1 = cv2.threshold(blurred_img, max_img - 2.9 * SD_img, 255, cv2.THRESH_BINARY)[1]

        # The next key step is to make the shape of the electrodes more clear. Need to flip the invert image again
        # otherwise eroded and dilate are all backwards... I could not flip at the beginning and redo the normalization
        # and take the lowest values but I m too lazy
        thresh_img1 = cv2.bitwise_not(thresh_img1)

        # create a nice ellipsoid kernel since we have round shape.
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # erode reduces noise around blobs. Makes the blobs smaller and more defined, especially when two are connected
        eroded_img = cv2.erode(thresh_img1, kernel_erode, iterations=3)

        # cv2.dilate increases the size of the blob, yield more defined dots
        dilated_img = cv2.dilate(eroded_img, kernel_erode, iterations=4)

        # The next is to perform a Connected-component labeling, which subsets connected components and labels them.
        dilated_img = np.where(cochlea_mask == 0, cochlea_mask, dilated_img)
        labels = measure.label(dilated_img, background=0)
        mask = np.zeros(eroded_img.shape, dtype="uint8")

        # utils.show(dilated_img)
        # store number of pixels per blob >> only used to find good limits for blob selection.
        numPixel_l = []

        # loop over all unique blobs that have been found
        for label in np.unique(labels):

            # Ignore the background label. Background = 0
            if label == 0:
                continue

            # Create mask and set labeled area "blob" to 255
            labelMask = np.zeros(eroded_img.shape, dtype="uint8")
            labelMask[labels == label] = 255

            # calculate numPixels
            numPixels = cv2.countNonZero(labelMask)

            # A normal electrode blob consists of around 1500 pixels on average. A weak blob should be at least 50, rest
            # is noise. These limits are based on empirical evidence we got from our image set. Would be greater to have
            # more to find more accurate values.
            if numPixels > 50 and numPixels < 3000:
                numPixel_l.append(numPixels)
                mask = cv2.add(mask, labelMask)

            # check iter depth. If a blob is below 50 it loops over although it should be eroded away with time and
            # adding a break to the elif on line 431 doesnt
            iter = 0

            # if a blob exceeds 3000 we assume that it is a collection of 2 or more blobs
            while numPixels > 3000 and iter != 100:
                # Activate this in case you want to see what happens also the utils.show() 3 lines lower
                # utils.show(labelMask)

                # erode the blob aggregate and recalculate blob size. If it is below 3000 it will make the cut and
                labelMask = cv2.erode(labelMask, kernel_erode, iterations=2)
                numPixels = cv2.countNonZero(labelMask)

                # see if blob aggregate is broken up after erosion
                # utils.show(labelMask1)

                # update counter
                iter += 1

                # included erorded blob if within interval [50, 3000]
                if numPixels > 50 and numPixels < 3000:
                    numPixel_l.append(numPixels)
                    mask = cv2.add(mask, labelMask)
                    break
                # if its too big or too small
                else:
                    numPixels = cv2.countNonZero(labelMask)
                    pass

        # here you can check that the average electrode blob size is around 1500
        print("mean blob size", np.mean(numPixel_l))

        mask = cv2.bitwise_not(mask)

        # check version of cv2 to load blob detector correctly
        if cv2.__version__.startswith('2.'):
            detector = cv2.SimpleBlobDetector(parameters)
        else:
            detector = cv2.SimpleBlobDetector_create(parameters)

        # detect the blobs
        blobs = detector.detect(mask)

        # draw red circles > np.array = temporary output array. Here its BGR not RGB
        drawBlobs = cv2.drawKeypoints(img, blobs, np.array([]), (255, 0, 0),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        all_coor = []
        # find the coordinates of the blobs
        for blob in blobs:
            cx = blob.pt[0]
            cy = blob.pt[1]
            all_coor.append((round(cx), round(cy)))

        utils.show(drawBlobs)

        return all_coor

    def setElectrodesCoordinates(self):

        verbose = config.set_electrode_coordinates["verbose"]
        # get post op image and invert intensity (ask Thibault, why the img is inverted in the first hand)
        img = self.postop_arr.copy()
        img = cv2.bitwise_not(img)

        # create mask of the cochlea for normalization. The cochlea is the region that is of interest to us, so we use
        # it to normalize the images. This region contains (most of the) electrodes as well as the background noise that
        # surrounds them.
        cochlea_mask = self.cochlea_area
        normRegion = np.where(cochlea_mask == 0, cochlea_mask, img)

        # Calculate mean and STD of normRegion for clipping and normalization
        mean, SD = cv2.meanStdDev(normRegion)

        # we clip the pixel values outside of the interval [mean-SD, mean+SD]
        clipped = np.clip(img, mean - 2 * SD, mean + 2 * SD).astype(np.uint8)

        # Normalize the image
        mean_cl, SD_cl = cv2.meanStdDev(clipped)
        img_norm = cv2.normalize(clipped, clipped, mean_cl, 255, norm_type=cv2.NORM_MINMAX)

        # Gaussian blurring to reduce high frequency noise
        blurred_img = cv2.GaussianBlur(img_norm, (41, 41), 0)

        # Extract statistics of the normalized image
        min_img, max_img = np.amin(blurred_img), np.amax(blurred_img)
        mean_img, SD_img = cv2.meanStdDev(blurred_img)

        # this might be overkill.. Here I set all values that are 2 SDs aways from the minimum to 0 (high pass fitlering)
        # blurred_img = np.where(blurred_img < min_img + 2*SD_img, 0, blurred_img)

        # thresholding of the images and conversion to a binary image
        # 3SD away from max value of image (= 255) seems reasonable
        thresh_img1 = cv2.threshold(blurred_img, max_img - 3 * SD_img, 255, cv2.THRESH_BINARY)[1]

        # The next key step is to make the shape of the electrodes more clear. Need to flip the invert image again
        # otherwise eroded and dilate are all backwards... I could not flip at the beginning and redo the normalization
        # and take the lowest values but I m too lazy
        thresh_img1 = cv2.bitwise_not(thresh_img1)

        # create a nice ellipsoid kernel since we have round shape.
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # erode reduces noise around blobs. Makes the blobs smaller and more defined, especially when two are connected
        eroded_img = cv2.erode(thresh_img1, kernel_erode, iterations=3)

        # cv2.dilate increases the size of the blob, yield more defined dots
        dilated_img = cv2.dilate(eroded_img, kernel_erode, iterations=2)

        # The next is to perform a Connected-component labeling, which subsets connected components and labels them.
        dilated_img = np.where(cochlea_mask == 0, cochlea_mask, dilated_img)
        labels = measure.label(dilated_img, background=0)
        mask = np.zeros(eroded_img.shape, dtype="uint8")

        # store number of pixels per blob >> only used to find good limits for blob selection.
        numPixel_l = []

        # loop over all unique blobs that have been found
        for label in np.unique(labels):

            # Ignore the background label. Background = 0
            if label == 0:
                continue

            # Create mask and set labeled area "blob" to 255
            labelMask = np.zeros(eroded_img.shape, dtype="uint8")
            labelMask[labels == label] = 255

            # calculate numPixels
            numPixels = cv2.countNonZero(labelMask)

            # A normal electrode blob consists of around 1500 pixels on average. A weak blob should be at least 50, rest
            # is noise. These limits are based on empirical evidence we got from our image set. Would be greater to have
            # more to find more accurate values.
            if numPixels > 50 and numPixels < 3000:
                numPixel_l.append(numPixels)
                mask = cv2.add(mask, labelMask)

            # check iter depth. If a blob is below 50 it loops over although it should be eroded away with time and
            # adding a break to the elif on line 431 doesnt
            # check iter depth. If a blob is below 50 it loops over although it should be eroded away with time and
            # adding a break to the elif on line 431 doesnt


            # if a blob exceeds 3000 we assume that it is a collection of 2 or more blobs
            while numPixels > 3000:
                # Activate this in case you want to see what happens also the utils.show() 3 lines lower
                # utils.show(labelMask)

                # erode the blob aggregate and recalculate blob size. If it is below 3000 it will make the cut and
                labelMask = cv2.erode(labelMask, kernel_erode, iterations=2)
                numPixels = cv2.countNonZero(labelMask)

                # see if blob aggregate is broken up after erosion
                # utils.show(labelMask1)



                # included erorded blob if within interval [50, 3000]
                if numPixels > 50 and numPixels < 3000:
                    numPixel_l.append(numPixels)
                    mask = cv2.add(mask, labelMask)
                    break
                # if its too big or too small
                else:
                    numPixels = cv2.countNonZero(labelMask)
                    pass

        # here you can check that the average electrode blob size is around 1500
        if verbose:
            print("\nAverage electrode blob size: ", np.mean(numPixel_l))

        # find the contours in the mask
        blob_contour = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blob_contour = imutils.grab_contours(blob_contour)

        # empty list to collect coordinates
        all_coor = []

        # add center coordinate
        center = self.cochlea_center
        cv2.circle(img, (int(center[0]), int(center[1])), 2, (0, 0, 0), 2)


        # loop over the contours
        for (i, c) in enumerate(blob_contour):
            # Highlight the potential electrode with a circle. Radius is not used..
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            all_coor.append((round(cX),round(cY)))

            # third argument = circle size, last argument = border thickness
            cv2.circle(img, (int(cX), int(cY)), 5, (255, 255, 255), 4)

        # Summary / Update
        print("\n{} potential electrodes have been found in {} \n".format(len(all_coor), self.preBasename[:-7]))

        # Show Image?
        if config.electrodes_enumeration["Show found electrodes on image?"]:
            name = self.name + " /  Detected electrode array"
            utils.show(img, name)

        if config.set_electrode_coordinates["Save Found Electrode File"]:
            print("saving file...")
            path2 = './GEN_IMG/'
            filename = "detected_electrodes_" + self.name + ".png"
            print("to path: ", path2 + filename)
            cv2.imwrite(os.path.join(path2, filename), img)

        # return list of coordinates
        return all_coor



    def setElectrodesSorted(self):
        """

        :return: list of tuples containing electrodes' (radius, angle, x, y)
        """
        center = self.cochlea_center
        points = self.electrodes_list

        ## TRANSFORM INTO POLAR FORM WITH FUNCTION polar()
        ## CREATE A LIST OF TUPLES WITH radius,angle,x,y
        sorting = [utils.polar(center, point) + point for point in points]

        ## SORT POINTS BY RADIUS (max to min) // TUPLE SORTING WITH KEY: LAMBDA
        sorted_angles = sorted(sorting, key=lambda tup: (-tup[0], tup[1]))

        ## IF MORE THAN 12 ELECTRODES ARE DETECTED, THE ONE'S FARTHEST AWAY ARE REMOVED (Less likely to be a hit)
        if len(points) > 12:
            sorted_angles = sorted_angles[len(points) - 12:]

        ## IF LESS ELECTRODES ARE FOUND, EMPTY SLOTS ARE ADDED SO THAT THE CORRECT ITERATION IS KEEP!
        ## ONE ADDITIONAL ITERATION IS ADDED TO FLUSH THE BUFFER AT THE END.

        if len(points) < 12:
            sorted_angles += [(0, 0, 0, 0)] * (12 - len(points))

        # sorted_angles = sorted(sorted_angles, reverse=True, key=lambda tup: tup[0])
        # print("SORTED ANGLES", len(sorted_angles), "\n", sorted_angles)

        return sorted_angles

    def setCochleaOrientation(self):
        """
        Check the rotation of spiral, given the coordinates of the furthest detected electrode.
        If this electrode is left and below of the center --> CLOCKWISE (returns 1)
        If the electrode is right and below the center --> COUNTER-CLOCKWISE (return -1)

        :return: None, if the detected electrode is above the center!
        """
        r_, theta_min, x_, y_ = self.relevant_electodes[0]
        CW = None

        if 180 > theta_min > 90:
            CW = 1
        elif 0 < theta_min <= 90:
            CW = -1
        elif CW is None:
            print("setCholeaOrientation failed!")
            print("Check if the center point and the first electrode are correctly identified")

        # print("CLOCKWISE", CW, "\n")
        return CW

    def setElectrodesOrder(self):
        electrodes = self.relevant_electodes
        CW = self.getCochleaOrientation()

        if CW is None:
            return 0

        electrodes_order = utils.electrode_sequence(CW, electrodes)
        # print("ORDER", len(electrodes_order), "\n", electrodes_order)
        return electrodes_order

    def setAngularInsertionDepth(self):
        """
        Returns the angular insertion depth for the detected electrodes in the proposed order (for the excel
        file.
        :return: List of tuples (electrode_i, x_cord, y_cord,theta_i)
        """
        sorted_electrodes = self.electrode_order
        CW = self.getCochleaOrientation()

        if CW is None:
            return 0

        # sorted_electrodes = sorted(sorted_electrodes, reverse=True, key=lambda tup: tup[0])
        ang_ins_depth = utils.calculate_angular_insertion_depth(sorted_electrodes, CW)
        # print("CORRECT SEQUENCE (electrode_i, x_cord, y_cord,theta_i):", len(ang_ins_depth), "\n", ang_ins_depth)

        return ang_ins_depth

    def enumerateElectrodes(self):
        """
        Show the output image with the electrodes annotated and the angular insertion depth shown.
        :return: nada
        """
        ang_ins_depth = self.getAngularInsertionDepth()
        image = self.getPostImg()
        image = cv2.bitwise_not(image)
        CW = self.getCochleaOrientation()
        name = self.name
        center = self.cochlea_center

        if CW is None:
            return 0

        cv2.circle(image,(int(center[0]),int(center[1])),2, (0, 0, 0), 2)
        for (i, c) in enumerate(ang_ins_depth):
            (el_number, cX, cY, theta_fin) = c

            # draw the bright spot on the image
            cX, cY = round(cX), round(cY)
            theta_fin = round(theta_fin)

            cv2.circle(image, (int(cX), int(cY)), 5, (255, 0, 0), 2)
            # cv2.putText(image, "#{}: {},{}".format(el_number,cX,cY), (cX - 60, cY - 10), cv2.FONT_HERSHEY_PLAIN, 1.45,(255, 0, 0),2)
            cv2.putText(image, "#{}: {}".format(el_number, theta_fin), (cX - 50, cY - 10), cv2.FONT_HERSHEY_PLAIN, 1.2,
                        (255, 0, 0), 2)

        utils.show(image, name)

        # Saving the image

        if config.electrodes_enumeration["save_file"]:
            print("saving file...")
            path2 = './GEN_IMG/'
            filename = "annotated_electrodes_" + self.name + ".png"
            print("to path: ", path2 + filename)
            cv2.imwrite(os.path.join(path2, filename), image)

        return 0

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
