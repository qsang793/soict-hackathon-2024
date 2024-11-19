"""Split data into train and val set"""

import os
import random
import shutil

from tqdm import tqdm


random.seed(0)

if __name__ == "__main__":
    data_root = r"D:\Project\SoICT2024\data\data_pool\nighttime\cam_03"
    image_dir = os.path.join(data_root, "images")
    label_dir = os.path.join(data_root, "labels")

    save_root = r"D:\Project\SoICT2024\data\data_final\cam_03"

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

    train_ratio = 0.9
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
