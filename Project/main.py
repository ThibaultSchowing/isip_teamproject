import utils
from Collection import Collection

print("ISIP - Team project 2020")

# Basic info
data_path = "./DATA"

# Import images by pre-post pair / function in utils.py
imgs_pair_paths = utils.get_img_pairs_paths(data_path)  # list of tuple / pairs of images pre-post
# pattern_path = ["./pattern/pattern.png", "./pattern/reversepattern.png", "./pattern/pattern.png", "./pattern/reversepattern.png"]
pattern_path = "./pattern/round2.png"
print(imgs_pair_paths)

# List of all images, not by pair
# all_imgs = list(sum(imgs_pair_paths, ()))

# Create a collection with all the pairs of picture
c = Collection(imgs_pair_paths, pattern_path)
pairs = c.getPairs()

############
for p in pairs:
    im = p.getCochleaCenter()
    print(im)

