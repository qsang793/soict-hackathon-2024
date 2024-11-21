import os
import shutil

import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


def infer_classify(model, source) -> int:
    classify_results = model.predict(
        source=source,
        verbose=False,
    )[0]
    cls_ = classify_results.probs.top1
    return cls_


if __name__ == "__main__":
    # Load model
    model_path = "/home/manhckv/manhckv/soict/weights/day_night/best.pt"
    model = YOLO(model_path, task="classify")
    classes = model.model.names

    # Warmup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_frame = np.zeros((640, 480, 3), dtype=np.uint8)
    model.predict(dummy_frame, device=device, verbose=False)

    # Output folders
    save_root = "__temp"
    os.makedirs(save_root, exist_ok=True)
    for cls_ in classes.values():
        os.makedirs(os.path.join(save_root, str(cls_)), exist_ok=True)

    # Input folder
    img_dir = "/home/manhckv/manhckv/soict/public_test"
    for img_name in tqdm(os.listdir(img_dir), desc="Inferencing classify"):
        img_path = os.path.join(img_dir, img_name)
        cls_ = infer_classify(model, img_path)
        shutil.copy(img_path, os.path.join(save_root, classes[cls_], img_name))
