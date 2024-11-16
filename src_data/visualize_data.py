import os

import cv2
from tqdm import tqdm

from utils.yolo_utils import convert_yolo_to_xyxy, read_yolo_txt, visualize_images


if __name__ == "__main__":
    data_root = "/home/manhckv/manhckv/soict/data/valid"
    images_dir = os.path.join(data_root, "images")
    labels_dir = os.path.join(data_root, "labels")

    num_to_visualize = 100  # if -1, visualize all images

    save_dir = "__visualized"
    os.makedirs(save_dir, exist_ok=True)

    for image_name in tqdm(os.listdir(images_dir)[:num_to_visualize]):
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, image_name.rsplit(".", 1)[0] + ".txt")

        image = cv2.imread(image_path)
        boxes, labels = read_yolo_txt(label_path)
        boxes = [convert_yolo_to_xyxy(box, image.shape[:2]) for box in boxes]

        image = visualize_images(image, boxes, labels)

        save_path = os.path.join(save_dir, image_name)
        cv2.imwrite(save_path, image)
