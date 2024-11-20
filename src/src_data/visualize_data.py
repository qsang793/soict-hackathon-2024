import os
import shutil

import cv2
from tqdm import tqdm

from utils.yolo_utils import convert_yolo_to_xyxy, read_yolo_txt, visualize_images


if __name__ == "__main__":
    data_root = r"D:\Project\SoICT2024\data\data_final\cam_01\train"
    image_dir = os.path.join(data_root, "images")
    label_dir = os.path.join(data_root, "labels")

    # Set to -1 to visualize all images
    num_to_visualize = 200

    save_dir = "__visualized"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    for image_name in tqdm(os.listdir(image_dir)[:num_to_visualize]):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))

        image = cv2.imread(image_path)
        boxes, labels = read_yolo_txt(label_path)
        boxes = [convert_yolo_to_xyxy(box, image.shape[:2]) for box in boxes]

        image = visualize_images(image, boxes, labels)

        save_path = os.path.join(save_dir, image_name)
        cv2.imwrite(save_path, image)
