"""This script applies vertical motion blur to vehicles in traffic images."""

import argparse
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm


CLASSES_MAP = {
    0: "motorbike",
    1: "car",
    2: "coach",
    3: "truck",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Motion Blur Script")
    parser.add_argument(
        "--data_root", type=str, help="Path to the data root with images and labels"
    )
    parser.add_argument(
        "--save_root", type=str, help="Path to folder to save the adjusted images"
    )
    return parser.parse_args()


def convert_yolo_to_xyxy(bbox, img_shape):
    image_h, image_w = img_shape[:2]
    xc, yc, w, h = bbox
    x1 = int((xc - w / 2) * image_w)
    y1 = int((yc - h / 2) * image_h)
    x2 = int((xc + w / 2) * image_w)
    y2 = int((yc + h / 2) * image_h)
    return x1, y1, x2, y2


def read_yolo_txt(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    boxes = []
    labels = []
    for line in lines:
        label, x, y, w, h = map(float, line.strip().split())
        boxes.append([x, y, w, h])
        if label in CLASSES_MAP:
            labels.append(int(label))
        else:
            print(f"Unknown class: {txt_path}, {label}")

    return boxes, labels


def calculate_blur_params(bbox, image_height=720):
    """
    Calculate blur parameters based on vehicle position relative to camera

    Args:
        bbox: Tuple of (x1, y1, x2, y2) in xyxy format
        image_height: Height of the image (720p)

    Returns:
        kernel_size: Size of the motion blur kernel
    """
    x1, y1, x2, y2 = bbox
    center_y = (y1 + y2) / 2

    # Calculate center point of bbox
    center_y = (y1 + y2) / 2

    # Invert the distance factor calculation
    # Now, objects at the top (far) will have distance_factor close to 0
    # Objects at the bottom (near) will have distance_factor close to 1
    distance_factor = center_y / image_height

    # Calculate kernel size based on distance
    min_kernel = 3
    max_kernel = 15
    kernel_size = int(min_kernel + (max_kernel - min_kernel) * distance_factor)

    # Make sure kernel size is odd
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

    return kernel_size


def create_vertical_motion_blur_kernel(kernel_size):
    """
    Create a vertical motion blur kernel
    """
    # Create a vertical kernel
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    kernel[:, center] = 1.0 / kernel_size
    return kernel


def augment_traffic_image(image, bboxes, labels):
    """
    Apply vertical motion blur to multiple vehicles in an image
    Only applies to bboxes larger than 100x100

    Args:
        image: Input image (1280x720)
        vehicle_bboxes: List of bboxes in xyxy format
    """
    result = image.copy()

    for bbox, label in zip(bboxes, labels):
        # Don't blur motorbikes
        if label == 0:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        if bbox_width * bbox_height > 5000:
            kernel_size = calculate_blur_params(bbox)
            kernel = create_vertical_motion_blur_kernel(kernel_size)

            roi = result[y1:y2, x1:x2]
            blurred_roi = cv2.filter2D(roi, -1, kernel)

            result[y1:y2, x1:x2] = blurred_roi

    return result


if __name__ == "__main__":
    args = parse_arguments()

    data_root = args.data_root
    image_dir = os.path.join(data_root, "images")
    label_dir = os.path.join(data_root, "labels")

    save_root = args.save_root
    save_image_dir = os.path.join(save_root, "images")
    save_label_dir = os.path.join(save_root, "labels")

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    for image_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))

        image = cv2.imread(image_path)
        bboxes, labels = read_yolo_txt(label_path)
        bboxes = [convert_yolo_to_xyxy(bbox, image.shape[:2]) for bbox in bboxes]

        augmented_image = augment_traffic_image(image, bboxes, labels)

        save_image_path = os.path.join(
            save_image_dir, image_name.replace(".jpg", "_blur.jpg")
        )
        save_label_path = os.path.join(
            save_label_dir, image_name.replace(".jpg", "_blur.txt")
        )

        cv2.imwrite(save_image_path, augmented_image)
        shutil.copy(label_path, save_label_path)
