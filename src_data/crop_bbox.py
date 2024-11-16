import os

import cv2
from tqdm import tqdm

from utils.const import CLASSES_MAP
from utils.yolo_utils import convert_yolo_to_xyxy, read_yolo_txt


if __name__ == "__main__":
    data_root = r"D:\Project\SoICT2024\data\train\nighttime"

    image_dir = os.path.join(data_root, "images")
    label_dir = os.path.join(data_root, "labels")

    save_root = "__temp/classify_vehicle_night"
    os.makedirs(save_root, exist_ok=True)

    for cls_ in CLASSES_MAP.values():
        os.makedirs(os.path.join(save_root, cls_), exist_ok=True)

    for image_name in tqdm(os.listdir(image_dir), desc="Processing images"):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.rsplit(".", 1)[0] + ".txt")

        image = cv2.imread(image_path)
        boxes, labels = read_yolo_txt(label_path)
        boxes = [convert_yolo_to_xyxy(box, image.shape[:2]) for box in boxes]

        for idx, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box
            cls_ = CLASSES_MAP[label]
            save_path = os.path.join(save_root, cls_, f"{image_name}_{idx}.jpg")
            try:
                cv2.imwrite(save_path, image[y1:y2, x1:x2])
            except Exception as e:
                print(f"Error when saving {save_path}: {e}")
                continue
