"""Split data into train and val set"""

import argparse
import os
import random
import shutil

from tqdm import tqdm


random.seed(0)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Split Data Script")
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Ratio of train set"
    )
    parser.add_argument(
        "--data_root", type=str, help="Path to the data root with images and labels"
    )
    parser.add_argument(
        "--save_root", type=str, help="Path to folder to save the adjusted images"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    data_root = args.data_root
    image_dir = os.path.join(data_root, "images")
    label_dir = os.path.join(data_root, "labels")

    save_root = args.save_root

    train_dir = os.path.join(save_root, "train")
    train_images_dir = os.path.join(train_dir, "images")
    train_labels_dir = os.path.join(train_dir, "labels")

    val_dir = os.path.join(save_root, "val")
    val_images_dir = os.path.join(val_dir, "images")
    val_labels_dir = os.path.join(val_dir, "labels")

    for dir_path in [
        save_root,
        train_dir,
        train_images_dir,
        train_labels_dir,
        val_dir,
        val_images_dir,
        val_labels_dir,
    ]:
        os.makedirs(dir_path, exist_ok=True)

    train_ratio = args.train_ratio
    list_images = os.listdir(image_dir)
    random.shuffle(list_images)

    for i, image_name in enumerate(tqdm(list_images, desc="Split data")):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))

        if i < len(list_images) * train_ratio:
            shutil.copy(image_path, train_images_dir)
            shutil.copy(label_path, train_labels_dir)
        else:
            shutil.copy(image_path, val_images_dir)
            shutil.copy(label_path, val_labels_dir)
