import config
import utils
from Collection import Collection

print("ISIP - Team project 2020")

# Basic info
data_directory = config.general["data_directory"]

# Import images by pre-post pair / function in utils.py

imgs_pair_paths = utils.get_img_pairs_paths(data_directory)  # list of tuple / pairs of images pre-post

pattern_path = config.pattern_matching_cochlea_center["pattern_path"]
print(imgs_pair_paths)

# List of all images, not by pair
# all_imgs = list(sum(imgs_pair_paths, ()))

# Create a collection with all the pairs of picture
c = Collection(imgs_pair_paths, pattern_path)

# Todo clear main - export csv (from class Collection)
############
for p in c.getPairs():
    im = p.getCochleaCenter()
    print(im)
