import os
import os.path

import matplotlib.pyplot as plt
import numpy as np


def get_img_pairs_paths(path=".\DATA"):
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
    print(name)
    print("Type: ", type(image))
    print("dType: ", image.dtype)
    print("Mean value: ", image.mean())


def show(img, name="Some image"):
    '''

    :param img:
    :param name:
    :return:
    '''
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
