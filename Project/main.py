from Collection import Collection
import utils
from skimage import io
from skimage.color import rgb2gray
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

print("ISIP - Team project 2020")

# Basic info
data_path = ".\\DATA"

# Import images by pre-post pair
imgs_pair_paths = utils.get_img_pairs_paths(data_path) # list of tuple / pairs of images pre-post
print(imgs_pair_paths)

# List of all images, not by pair
# all_imgs = list(sum(imgs_pair_paths, ()))

c = Collection(imgs_pair_paths)
pairs = c.getPairs()
p = pairs[1]

im = p.getSpiralCenter()

#plt.figure()
#plt.hist(im.flatten(),128)
#plt.show()
print(type(im))
print(im.shape)

m = np.where(np.logical_or(im > 100, im < 50))
im[m] = 0
utils.show(im)

# for img in all_imgs:
#     print(img)
#     i = io.imread(img)
#     print(i.shape)
#     j = rgb2gray(i)
#     print(type(j[0][0]))
#     j *= 255 / j.max()
#     print(j.shape, " - ", np.amax(j), " : ", np.amin(j))



