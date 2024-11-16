import os
import shutil

from tqdm import tqdm
from ultralytics import YOLO

from utils.yolo_utils import CLASSES_MAP


def infer_classify(source, conf) -> int:
    classify_results = model.predict(
        source=source,
        conf=conf,
        verbose=False,
    )[0]

    cls_ = classify_results.probs.top1

    return cls_


if __name__ == "__main__":
    # Input folder
    img_dir = "data/classify_vehicle/train/coach"

    # Output folders
    save_root = "__temp"
    os.makedirs(save_root, exist_ok=True)
    for cls_ in CLASSES_MAP.values():
        os.makedirs(os.path.join(save_root, cls_), exist_ok=True)

    # Load model
    model_path = "SOICT2024-VEHICLE-CLASSIFICATION/run_0/weights/best.pt"
    model = YOLO(model_path, task="classify")
    CLS_FIX = {
        0: "car",
        1: "coach",
        2: "motorbike",
        3: "truck",
    }

    conf = 0.9

    for img_name in tqdm(os.listdir(img_dir), desc="Inferencing classify"):
        img_path = os.path.join(img_dir, img_name)
        cls_ = infer_classify(img_path, conf)
        shutil.copy(img_path, os.path.join(save_root, CLS_FIX[cls_], img_name))
