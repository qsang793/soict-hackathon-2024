"""This script is used to split the dataset to camera level."""

import os
import shutil


def parse_filename(image_dir):
    all_filename = os.listdir(image_dir)
    unique_prefixes = set()

    for img_name in all_filename:
        prefix = img_name.split("_")[:2]
        unique_prefixes.add("_".join(prefix))

    ## get list file name by prefix
    file_per_prefix = {}
    for prefix in unique_prefixes:
        file_per_prefix[prefix] = []
        for img_name in all_filename:
            if img_name.startswith(prefix):
                file_per_prefix[prefix].append(img_name)

    ## print number of images by prefix
    for prefix in unique_prefixes:
        print(f"{prefix}: {len(file_per_prefix[prefix])} images")

    return file_per_prefix


if __name__ == "__main__":
    data_root = r"C:\Users\caoma\Downloads\aihcmc\aic_hcmc2020\aic_hcmc2020"
    image_dir = os.path.join(data_root, "images")
    label_dir = os.path.join(data_root, "labels")

    save_root = r"D:\Project\SoICT2024\data\data_pool\aihcmc"
    img_per_prefix = parse_filename(image_dir)

    for prefix, img_list in img_per_prefix.items():
        prefix_dir = os.path.join(save_root, prefix)
        os.makedirs(prefix_dir, exist_ok=True)
        os.makedirs(os.path.join(prefix_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(prefix_dir, "labels"), exist_ok=True)

        for img_name in img_list:
            shutil.copy(
                os.path.join(image_dir, img_name), os.path.join(prefix_dir, "images")
            )
            shutil.copy(
                os.path.join(label_dir, img_name.replace(".jpg", ".txt")),
                os.path.join(prefix_dir, "labels"),
            )
