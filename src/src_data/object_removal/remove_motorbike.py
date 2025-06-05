"""This script removes motorbike from images and labels"""

import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.src_data.object_removal.core import process_inpaint
from src.utils.yolo_utils import convert_yolo_to_xyxy, read_yolo_txt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Remove Motorbike Script")
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
    save_image_dir = os.path.join(save_root, "images")
    save_label_dir = os.path.join(save_root, "labels")
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    for image_name in tqdm(os.listdir(image_dir), desc="Processing images"):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.rsplit(".", 1)[0] + ".txt")

        input_image = Image.open(image_path)
        width, height = input_image.size

        xywhn_boxes, labels = read_yolo_txt(label_path)
        num_motorbike = np.sum(np.array(labels) == 0)
        num_other = len(labels) - num_motorbike

        # Skip images with less than 5 motorbikes or 2 other objects
        if num_motorbike < 5 or num_other < 2:
            print(f"Skip {image_name}")
            continue

        xyxy_boxes = [convert_yolo_to_xyxy(box, (width, height)) for box in xywhn_boxes]

        mask = Image.new("RGBA", (width, height), (0, 0, 0, 255))

        new_label = []
        for xyxy_box, xywhn_box, label in zip(xyxy_boxes, xywhn_boxes, labels):
            if label != 0:
                new_label.append(
                    f"{label} {xywhn_box[0]} {xywhn_box[1]} {xywhn_box[2]} {xywhn_box[3]}\n"
                )
                continue

            x1, y1, x2, y2 = xyxy_box
            w = x2 - x1
            h = y2 - y1
            x1 = max(0, x1 - w // 10)
            y1 = max(0, y1 - h // 10)
            x2 = min(width, x2 + w // 10)
            y2 = min(height, y2 + h // 10)

            mask.paste((0, 0, 0, 0), (x1, y1, x2, y2))

        output = process_inpaint(np.array(input_image), np.array(mask))
        img_output = Image.fromarray(output).convert("RGB")
        img_output.save(os.path.join(save_image_dir, image_name))

        with open(
            os.path.join(save_label_dir, image_name.rsplit(".", 1)[0] + ".txt"), "w"
        ) as f:
            f.writelines(new_label)
