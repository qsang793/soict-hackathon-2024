import os
import random

import cv2
import numpy as np


def load_labels(label_path):
    """Load bounding boxes from label file."""
    with open(label_path, "r") as f:
        labels = [line.strip().split() for line in f]
    return [
        [int(l[0]), float(l[1]), float(l[2]), float(l[3]), float(l[4])] for l in labels
    ]


def save_labels(label_path, labels):
    """Save updated bounding boxes to label file."""
    with open(label_path, "w") as f:
        for label in labels:
            f.write(
                f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n"
            )


def check_overlap(bbox1, bbox2, overlap_threshold):
    """Check if two bounding boxes overlap too much."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Calculate the intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate the area of both bounding boxes
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate the overlap ratio
    overlap_ratio = inter_area / min(bbox1_area, bbox2_area)

    return overlap_ratio > overlap_threshold


def augment_image_with_label(
    image_path,
    label_path,
    sample_images,
    output_dir,
    target_labels,
    overlap_threshold=0.5,
):
    """
    Augment image by pasting sample images for missing target labels using a random placement approach.

    Args:
        image_path (str): Path to the original image.
        label_path (str): Path to the original label file.
        sample_images (dict): Dictionary containing sample images for each label.
        output_dir (str): Directory to save the augmented image and label.
        target_labels (list): List of labels to ensure each image has.
        overlap_threshold (float): Maximum allowed overlap ratio between bounding boxes.
    """
    # Load the image and label data
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    labels = load_labels(label_path)

    # Calculate the average number of images per class
    label_counts = {label: 0 for label in target_labels}
    for label in labels:
        if label[0] in label_counts:
            label_counts[label[0]] += 1
    avg_images = np.ceil(sum(label_counts.values()) / len(target_labels))

    # Identify how many images to add for each label
    images_to_add = {
        label: max(0, int(avg_images - count)) for label, count in label_counts.items()
    }

    # Loop over each label and add the required number of sample images
    for label, count in images_to_add.items():
        for _ in range(count):
            sample_imgs = sample_images[label]
            placed = False
            for _ in range(len(sample_imgs)):
                sample_img = random.choice(sample_imgs)
                sh, sw, _ = sample_img.shape  # Sample image dimensions

                # Try random positions until a suitable one is found
                for _ in range(100):  # Try up to 100 times
                    x = random.randint(0, w - sw)
                    y = random.randint(0, h - sh)

                    # Calculate the bounding box of the sample image
                    new_bbox = [x, y, x + sw, y + sh]

                    # Check for overlap with existing bounding boxes
                    overlap = False
                    for existing_label in labels:
                        existing_bbox = [
                            int((existing_label[1] - existing_label[3] / 2) * w),
                            int((existing_label[2] - existing_label[4] / 2) * h),
                            int((existing_label[1] + existing_label[3] / 2) * w),
                            int((existing_label[2] + existing_label[4] / 2) * h),
                        ]
                        if check_overlap(new_bbox, existing_bbox, overlap_threshold):
                            overlap = True
                            break

                    if not overlap:
                        # Place the sample image onto the original image
                        image[y : y + sh, x : x + sw] = sample_img

                        # Calculate normalized coordinates for the bounding box
                        new_bbox_normalized = [
                            x / w + sw / (2 * w),
                            y / h + sh / (2 * h),
                            sw / w,
                            sh / h,
                        ]
                        labels.append(
                            [
                                label,
                                new_bbox_normalized[0],
                                new_bbox_normalized[1],
                                new_bbox_normalized[2],
                                new_bbox_normalized[3],
                            ]
                        )

                        placed = True
                        break

                if placed:
                    break

    # Save augmented image and updated labels
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    output_label_path = os.path.join(output_dir, os.path.basename(label_path))
    cv2.imwrite(output_image_path, image)
    save_labels(output_label_path, labels)


# Example usage
sample_image_dirs = {
    0: "D:/2_Hackathon/data_new/sample_object/class_1/night",
    1: "D:/2_Hackathon/data_new/sample_object/class_2/night",
    2: "D:/2_Hackathon/data_new/sample_object/class_3/night",
    3: "D:/2_Hackathon/data_new/sample_object/class_4/night",
}

# Load sample images from directories
sample_images = {}
for label, dir_path in sample_image_dirs.items():
    sample_images[label] = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        img = cv2.imread(file_path)
        if img is not None:
            sample_images[label].append(img)

image_folder = "D:/2_Hackathon/data_new/nighttime_normalized"
label_folder = "D:/2_Hackathon/data_new/nighttime_normalized"
output_dir = "augmented_data"
os.makedirs(output_dir, exist_ok=True)

# List all images and labels
for image_file in os.listdir(image_folder):
    if image_file.endswith(".jpg"):
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(image_folder, image_file.replace(".jpg", ".txt"))
        augment_image_with_label(
            image_path,
            label_path,
            sample_images,
            output_dir,
            target_labels=[0, 1, 2, 3],
            overlap_threshold=0.1,
        )
