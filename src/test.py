import os

import cv2


img_dir = "/home/manhckv/manhckv/soict/data/data-4cls/full/train/images"
# img_dir = "/home/manhckv/manhckv/soict/data/public_test"

all_imgs = os.listdir(img_dir)

unique_prefixes = set()

img_sizes = []
for img_name in all_imgs:
    prefix = img_name.split("_")[:2]
    unique_prefixes.add("_".join(prefix))

    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    img_sizes.append(img.shape)

print(unique_prefixes)

print(set(img_sizes))
