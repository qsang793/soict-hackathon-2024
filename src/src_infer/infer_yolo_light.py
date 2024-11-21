import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.LoLi_IEA.LoLi_IEA import LoLi_IEA
from src.utils.yolo_utils import visualize_images


def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    # parser.add_argument("--img_dir", type=str, default="public_test")
    parser.add_argument(
        "--img_dir", type=str, default="/home/manhckv/manhckv/soict/__visualized"
    )
    parser.add_argument(
        "--vehicle_model_path", type=str, default="weights/vehicle/epoch100.pt"
    )
    parser.add_argument(
        "--daynight_model_path", type=str, default="weights/day_night/best.pt"
    )
    parser.add_argument("--loli_iea_model_dir", type=str, default="weights/LoLi_IEA")

    parser.add_argument("--output_path", type=str, default="predict.txt")
    parser.add_argument(
        "--vis", action="store_true", help="Visualize the images with detections"
    )
    return parser.parse_args()


def infer_classify(model, source) -> int:
    classify_results = model.predict(
        source=source,
        verbose=False,
    )[0]
    cls_ = classify_results.probs.top1
    return cls_


def infer_detect(model, source, conf=0.01, iou=1):
    detections = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        verbose=False,
    )[0]
    return detections


if __name__ == "__main__":
    ## Parse arguments ----------------------------------------------
    args = parse_arguments()

    if args.vis:
        visualized_dir = "__visualized"
        os.makedirs(visualized_dir, exist_ok=True)

    img_dir = args.img_dir
    output_path = args.output_path

    ## Load a model -------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vehicle_model_path = args.vehicle_model_path
    vehicle_model = YOLO(vehicle_model_path, task="detect")

    daynight_model_path = args.daynight_model_path
    daynight_model = YOLO(daynight_model_path, task="classify")

    light_enhancer = LoLi_IEA(args.loli_iea_model_dir, device)

    ## Warmup -------------------------------------------------------
    dummy_frame = np.zeros((640, 480, 3), dtype=np.uint8)
    vehicle_model.predict(dummy_frame, device=device, verbose=False)
    daynight_model.predict(dummy_frame, device=device, verbose=False)

    ## Inferencing ---------------------------------------------------
    results = []
    for img_name in tqdm(os.listdir(img_dir), desc="Inferencing"):
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)

        day_night_cls = infer_classify(daynight_model, image)
        if day_night_cls == 1:
            image = light_enhancer.enhance_image(image)

        detections = infer_detect(vehicle_model, image)

        if len(detections.boxes) == 0:
            continue

        boxes = detections.boxes.xyxy.cpu().numpy()
        boxes_xywhn = detections.boxes.xywhn.cpu().numpy()
        labels = detections.boxes.cls.cpu().tolist()
        scores = detections.boxes.conf.cpu().tolist()

        for box, label, score in zip(boxes_xywhn, labels, scores):
            result = (
                f"{img_name} {int(label)} {box[0]} {box[1]} {box[2]} {box[3]} {score}"
            )
            results.append(result)

        if args.vis:
            image = cv2.imread(img_path)
            visualized_img = visualize_images(image, boxes, labels)
            save_path = os.path.join(visualized_dir, img_name)
            cv2.imwrite(save_path, visualized_img)

    ## Save results to file
    with open(output_path, "w") as f:
        for result in results:
            f.write(result + "\n")
