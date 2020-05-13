import os, os.path
import matplotlib.pyplot as plt
from PIL.Image import Image


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
    plt.imshow(img)
    plt.axis('off')
    plt.title(name)
    plt.show()





# https://www.oreilly.com/library/view/programming-computer-vision/9781449341916/ch01.html