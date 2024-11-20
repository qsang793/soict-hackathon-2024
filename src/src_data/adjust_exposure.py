"""This script is used to adjust the exposure by applying gamma correction."""

import os
import random
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def adjust_gamma(image, gamma):
    # Build a lookup table mapping the pixel values
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table), gamma


if __name__ == "__main__":
    data_root = r"D:\Project\SoICT2024\data\data_final\cam_10\train"
    image_dir = os.path.join(data_root, "images")
    label_dir = os.path.join(data_root, "labels")

    save_root = r"D:\Project\SoICT2024\data\data_final\cam_10\train_gamma"
    save_image_dir = os.path.join(save_root, "images")
    save_label_dir = os.path.join(save_root, "labels")

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    for image_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))

        image = cv2.imread(image_path)

        adjusted, gamma_values = adjust_gamma(image, gamma=random.uniform(0.2, 0.6))

        save_image_path = os.path.join(
            save_image_dir, image_name.replace(".jpg", "_gamma.jpg")
        )

        save_label_path = os.path.join(
            save_label_dir, image_name.replace(".jpg", "_gamma.txt")
        )

        cv2.imwrite(save_image_path, adjusted)
        shutil.copy(label_path, save_label_path)
