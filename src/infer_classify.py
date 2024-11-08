import os
import shutil

from tqdm import tqdm
from ultralytics import YOLO


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
    img_dir = "/home/manhckv/manhckv/soict/data/public_test"

    # Output folders
    day_folder = "__temp/day"
    night_folder = "__temp/night"
    os.makedirs(day_folder, exist_ok=True)
    os.makedirs(night_folder, exist_ok=True)

    # Load model
    model_path = "SOICT2024-VEHICLE-CLASSIFICATION/run_0/weights/best.pt"
    model = YOLO(model_path, task="classify")

    conf = 0.9

    for img_name in tqdm(os.listdir(img_dir), desc="Inferencing classify"):
        img_path = os.path.join(img_dir, img_name)
        cls_ = infer_classify(img_path, conf)
        if cls_ == 0:
            shutil.copy(img_path, day_folder)
        else:
            shutil.copy(img_path, night_folder)
