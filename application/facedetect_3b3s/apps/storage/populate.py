import os
import os.path
import random
from glob import glob
from shutil import copy

random.seed(123)

datapath = "../../../data"
storagepath = "store"

# Extract the images, and the image numbers
image_paths = []
for dir,_,_ in os.walk(datapath):
    image_paths.extend(glob(os.path.join(dir, "*.jpg")))
image_paths.sort(key=(lambda s: int(s.split("/")[-1].split("_")[-1].split(".")[0])))

img_nbr = list(set([int(s.split("/")[-1].split("_")[-1].split(".")[0]) for s in image_paths]))
img_nbr.sort()

# Create a dictionary, where each element is a list of paths to images with the same name
images_binned = {}
for i in img_nbr:
    images_binned[i] = []

for img_path in image_paths:
    i = int(img_path.split("/")[-1].split("_")[-1].split(".")[0])
    images_binned[i].append(img_path)

# Remove old files
old_files = glob(storagepath + "/*.jpg")
for f in old_files:
    os.remove(f)

# Copy new files
for i in img_nbr:
    copy(random.choice(images_binned[i]), storagepath)


