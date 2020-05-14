import os, os.path
import matplotlib.pyplot as plt
from PIL.Image import Image
import cv2
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


def import_images(paths_list, verbose=False):

    for img in paths_list:

        # Open images as <class 'PIL.Image.Image'> with values between 0 and 255
        i = Image.open(img).convert('L')

        if(verbose):
            print(img)
            print(i.size)
            print(i.getextrema())
            print(type(i))
    return

def show(img, name = "Some image"):
    #plt.subplot(1,1,1)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.axis('off')
    plt.title(name)
    plt.show()

# match a template on a picture and return the best result and the coordinates
# returns: (center list, method)
def best_match_location(im, template):
    w, h = template.shape[::-1]

    # results
    score_list = []
    center_list = []

    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF']#, 'cv2.TM_SQDIFF_NORMED']

    # Do it for each method
    for meth in methods:
        print("##############################")
        print("Method: ", meth)
        img = im.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img, template, method)
        # Normalized Data between 0 and 1
        print("Result type: ", type(res))
        print("result shape: ", res.shape)
        print("result values: ", np.amin(res), " - ", np.amax(res))

        norm = np.linalg.norm(res) # https://kite.com/python/answers/how-to-normalize-an-array-in-numpy-in-python
        normalized = (res/norm) * 255
        print("result norm values: ", np.amin(normalized), " - ", np.amax(normalized))

        score_list.append(normalized)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED: reverse normalized picture
        # -> min value (highest correlation) becomes highest
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            res = (1 - res)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(normalized)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        center_list.append(((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2))
    return (center_list, methods, score_list)


# https://www.oreilly.com/library/view/programming-computer-vision/9781449341916/ch01.html