import os
import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

import config


def normalize_2dimage_grayscale(image):
    """
    Normalizes an image between 0 and 255
    :param image:
    :return:
    """
    # image = cv2.blur(image, (config.preprocessing_1["blur"], config.preprocessing_1["blur"]))
    normalizedImg = np.zeros(image.shape)
    normalizedImg = cv2.normalize(image, normalizedImg, 0, 255, cv2.NORM_MINMAX)
    return normalizedImg


# mainly just ideas and examples of useful functions to pre-process images
def preprocess_image_grayscale(image, name="Preprocessed image"):
    """
    Preprocess an image / just a bunch of usefull functions that might come handy
    :param image: Image to process / grayscale 0-255
    :param name:
    :return:
    """
    pimage = image.copy()

    # Noise reduction
    # Strong blur, very efficient (65, 65)
    img_blur = cv2.blur(pimage, (config.preprocessing_1["blur"], config.preprocessing_1["blur"]))

    img_threshold = np.where(np.logical_or(img_blur > config.preprocessing_1["thr_up_gray"],
                                           img_blur < config.preprocessing_1["thr_low_gray"]),
                             255,
                             0).astype(np.uint8)

    edged = cv2.Canny(img_threshold,
                      config.preprocessing_1["canny_thr1"],
                      config.preprocessing_1["canny_thr2"])

    #  fill in the holes between edges with dilation
    dilated_image = cv2.dilate(edged, np.ones((5, 5)))

    return dilated_image


def get_img_pairs_paths(path=config.general["data_directory"], sep="\\"):
    """

    :param sep: this is the appropriate separator for creating a string of the paths of the CT scans.
    :param path: Path to the directory containing the data. Works according to the given structure. (not robust)
    :return: List of tuples containing the paths of the pre-surgery and post-surgery CT scans.
    """
    pairs = []
    for root, dirs, files in os.walk(path):
        pair = []

        if config.general["Running on MacOS?"]:
            sep = "/"

        for file in files:
            pair.append(root + sep + file)
        pairs.append(tuple(pair))

    # Removes the ./DATA entry
    del pairs[0]
    return pairs


def get_image_info(image, name="Image Info"):
    """
    Debug function to display image information
    :param image:
    :param name:
    :return: Nada, just print stuff.
    """
    print(name)
    print("Type: ", type(image))
    print("Shape: ", image.shape)
    print("dType: ", image.dtype)
    print("Mean value: ", image.mean())

    return None


def show(img, name="Some image"):
    """

    :param img:
    :param name:
    :return:
    """
    # plt.subplot(1,1,1)
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.axis('off')
    plt.title(name)
    plt.show()


def create_circular_mask(h, w, center=None, radius=None):
    '''
    Creates a binary image of size h x w with a circle
    :param h: height of image
    :param w: width of image
    :param center: center of circle
    :param radius: radius of circle
    :return:
    '''
    # https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


# https://www.oreilly.com/library/view/programming-computer-vision/9781449341916/ch01.html

#### FUNCTIONS FOR NUMBERING ELECTRODES AND CALCULATING THE INSERTION DEPTH ANGLE

def polar(center, electrode):
    """
    A converter from carthesian to polar cordinates for a point, given a specific center
    :param center: takes the x,y cord. of the center of the spiral
    :param electrode: takes the x,y cord of the point (electrode)
    :return r, theta: returns the polar cordinates (r and theta in degrees) of the electrode
    """
    x, y = electrode
    x0, y0 = center

    xc, yc = x - x0, y - y0

    if xc == 0:
        xc += sys.float_info.epsilon

    theta = math.atan(yc / xc) * 180 / math.pi
    r = math.sqrt(xc ** 2 + yc ** 2)

    if xc < 0:
        theta += 180
    elif xc > 0 and yc < 0:
        theta += 360

    return round(r, 2), round(theta, 2)


def electrode_sequence(CW, sorted_angles):
    """
    This function determines the correct sequence of how the electrodes are assembled in the spiral shape, when
    the spiral is unwound.
    :param CW: This determines whether the spiral is turned clockwise CW = 1 or counter-clockwise CW = -1.
    :param sorted_angles: It takes a list of tuples (radius, angle) off all electrodes. The list is (initially) sorted
    by the length of radius (--> tup[0]).
    :return: returns the correct sequence of the electrodes as a list with tuple (radius, angle).
    """
    ## THE SEQUENCE IS WHERE THE CORRECT ORDER IS BUILT, WHILE THE BUFFER LIST HOLDS THE ELECTRODES UNTIL THEY ARE
    ## THE BEST FIT FOR THE NEXT ELECTRODE IN THE SEQUENCE.
    sequence, buffer = [], []
    set_sorted_angles = sorted_angles.copy()
    max_angle, min_angle = config.electrodes_enumeration["max_angle"], config.electrodes_enumeration["min_angle"]
    r_thresh = config.electrodes_enumeration["Radius threshold"]

    ## THE OUTERMOST ELECTRODE
    r_1, theta_1, x_1, y_1 = sorted_angles.pop(0)

    sequence.append((r_1, theta_1, x_1, y_1))

    for i in range(len(set_sorted_angles) - 1):
        r_i, theta_i, x_i, y_i = set_sorted_angles[i + 1]
        # print("VALUE %s\n" % str(i + 2), r_i, theta_i, x_i, " ", y_i, sep="\t")

        ## CHECK BUFFER FOR MATCH:
        Loop = True
        k, l = 0, 0
        if len(buffer) > 0:
            while Loop:
                for j in range(len(buffer)):

                    if len(buffer) == 0 or len(buffer) >= j+k:
                        Loop = False

                    r_j, theta_j, x_j, y_j = buffer[j - k]
                    # print("B CHECK:", max_angle > CW * (theta_1 - theta_j) > min_angle, r_thresh[1] * r_1 > r_j >
                    # r_thresh[0] * r_1, -(360 - min_angle) < CW * (theta_1 - theta_j) < -(360 - max_angle),
                    # r_thresh[1] * r_1 > r_j > r_thresh[0] * r_1)

                    if max_angle > CW * (theta_1 - theta_j) > min_angle and r_thresh[1] * r_1 > r_j > r_thresh[0] * r_1 \
                            or -(360 - min_angle) < CW * (theta_1 - theta_j) < -(360 - max_angle) \
                            and r_thresh[1] * r_1 > r_j > r_thresh[0] * r_1:
                        del buffer[j - k]
                        k += 1
                        l += 1
                        sequence.append((r_j, theta_j, x_j, y_j))
                        r_1, theta_1, x_1, y_1 = r_j, theta_j, x_j, y_j

                    if l == 0:
                        Loop = False
                # print("l",l, len(buffer))
                if l <= 0 or len(buffer) == 0:
                    Loop = False

                l -= 1

        # print("S CHECK: ", 60 >= CW * (theta_1 - theta_i) >= 0,r_thresh[1] * r_1 >= r_i >= r_thresh[0] * r_1,
        # 180 < CW * (theta_1 - theta_i) <= 270,r_thresh[1] * r_1 > r_i > r_thresh[0] * r_1)
        if (x_i, y_i) == (0, 0):
            continue
        elif 60 >= CW * (theta_1 - theta_i) >= 0 and r_thresh[1] * r_1 >= r_i >= r_thresh[0] * r_1 \
                or 180 < CW * (theta_1 - theta_i) <= 270 and r_thresh[1] * r_1 > r_i > r_thresh[0] * r_1:
            # print("added Value {} to seq".format(i + 2))
            sorted_angles.pop(0)
            sequence.append((r_i, theta_i, x_i, y_i))
            r_1, theta_1, x_1, y_1 = r_i, theta_i, x_i, y_i
        else:
            # print("added Value {} to buffer".format(i + 2))
            buffer.append((r_i, theta_i, x_i, y_i))
            buffer.sort(reverse=False, key=lambda tup: tup[1])

    # print("Value before\n", r_1, theta_1, x_1, y_1, sep="\t")
    # print("SEQ", len(sequence), "\n", sequence)

    # print("BUFFER:", len(buffer), "\n", buffer)
    return sequence


def calculate_angular_insertion_depth(sorted_angle, CW):
    """
    Calculates the angular insertion depth of each electrode given the correct sequence.
    :param sorted_angle: this is the list of tuples (radius, angle, x,y) of each electrode in the correct order.

    :param CW: The rotation of spiral. Either clockwise, 1 or counter clockwise -1.
    :return: A list of tuples (electrode nr., x-cord., y-cord., angular insertion depth)
    """
    verbose = config.calculate_angular_insertion_depth["verbose"]

    ins_depth = [(0, 0, 0, 0)] * len(sorted_angle)  # or * 12
    r, theta_zero, x_zero, y_zero = sorted_angle[0]
    theta_tot = 0

    electrode_nr = 12
    for (i, ind) in enumerate(sorted_angle):
        r_i, theta_i, x_i, y_i = ind
        theta_diff = CW * (theta_zero - theta_i)
        if verbose:
            print("BEFORE", theta_diff, theta_diff > 270, theta_diff < 0)

        if theta_diff <= - (360 - config.electrodes_enumeration["max_angle"]):
            theta_diff = 360 + theta_diff

        theta_tot += theta_diff

        if verbose:
            print("electrode", electrode_nr - i, ":", theta_diff, theta_tot)

        ins_depth[i] = (electrode_nr - i, x_i, y_i, round(theta_tot, 5))

        theta_zero = theta_i

    # ins_depth = sorted(ins_depth, key=lambda tup: tup[0])

    return ins_depth
