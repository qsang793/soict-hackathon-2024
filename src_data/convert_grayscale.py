"""This script converts images to grayscale"""

import os
import shutil

import cv2
from tqdm import tqdm


def convert_to_grayscale(image_path):
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image


if __name__ == "__main__":
    data_root = r"D:\Project\SoICT2024\data\data_final\cam_03\val"
    image_dir = os.path.join(data_root, "images")
    label_dir = os.path.join(data_root, "labels")

    save_root = r"D:\Project\SoICT2024\data\data_final\cam_03\val_grayscale"
    save_image_dir = os.path.join(save_root, "images")
    save_label_dir = os.path.join(save_root, "labels")

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    for image_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))

        adjusted = convert_to_grayscale(image_path)

        save_image_path = os.path.join(
            save_image_dir, image_name.replace(".jpg", "_grayscale.jpg")
        )

        save_label_path = os.path.join(
            save_label_dir, image_name.replace(".jpg", "_grayscale.txt")
        )

        cv2.imwrite(save_image_path, adjusted)
        shutil.copy(label_path, save_label_path)
