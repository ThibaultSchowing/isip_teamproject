import os
import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np

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


def get_img_pairs_paths(path=".\DATA"):
    """

    :param path: Path to the directory containing the data. Works according to the given structure. (not robust)
    :return: List of tuples containing the paths of the pre-surgery and post-surgery CT scans.
    """
    pairs = []
    for root, dirs, files in os.walk(path):
        pair = []
        for file in files:
            pair.append(root + "\\" + file)
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
