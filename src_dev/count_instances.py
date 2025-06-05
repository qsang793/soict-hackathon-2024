import os

from src.utils.yolo_utils import CLASSES_MAP, read_yolo_txt


root_dir = "/home/manhckv/vm7608/soict/__data/test/all"

label_dir = os.path.join(root_dir, "labels")

count = {
    "motorbike": 0,
    "car": 0,
    "coach": 0,
    "truck": 0,
}

for label_file in os.listdir(label_dir):
    label_path = os.path.join(label_dir, label_file)
    boxes, labels = read_yolo_txt(label_path)

    for label in labels:
        if label in CLASSES_MAP:
            count[CLASSES_MAP[label]] += 1
        else:
            print(f"Unknown class: {label_file}, {label}")

print(count)
