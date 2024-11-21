import argparse
import os
from dataclasses import dataclass

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from src.utils.yolo_utils import visualize_images


@dataclass
class Config:
    detect_conf_thres: float = 0.01
    detect_iou_thres: float = 0.7


def infer_classify(source, model) -> int:
    classify_results = model.predict(
        source=source,
        verbose=False,
    )[0]

    cls_ = classify_results.probs.top1
    # score = classify_results.probs.top1conf
    # cls_fix = {
    #     0: 1,
    #     1: 2,
    #     2: 0,
    #     3: 3,
    # }

    # return cls_fix[cls_], score
    return cls_


def infer_detect(source, model):
    detections = model.predict(
        source=source,
        conf=Config.detect_conf_thres,
        iou=Config.detect_iou_thres,
        verbose=False,
    )[0]

    return detections


def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    parser.add_argument(
        "--vis", action="store_true", help="Visualize the images with detections"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.vis:
        visualized_dir = "__visualized"
        os.makedirs(visualized_dir, exist_ok=True)

    img_dir = "/home/manhckv/manhckv/soict/data/public_test"
    output_path = "predict.txt"

    # Load a model
    det_model_path = "SOICT2024-VEHICLE-DETECTION-1CLS/run_2/weights/best.pt"
    det_model = YOLO(det_model_path)

    cls_model_path = "SOICT2024-VEHICLE-CLASSIFICATION/run_1/weights/best.pt"
    cls_model = YOLO(cls_model_path, task="classify")

    results = []
    for img_name in tqdm(os.listdir(img_dir), desc="Inferencing"):
        img_path = os.path.join(img_dir, img_name)

        detections = infer_detect(img_path, det_model)

        if len(detections.boxes) == 0:
            continue

        boxes = detections.boxes.xyxy.cpu().numpy()
        boxes_xywhn = detections.boxes.xywhn.cpu().numpy()
        scores = detections.boxes.conf.cpu().tolist()

        labels = []
        for box, box_xywhn, score in zip(boxes, boxes_xywhn, scores):
            x1, y1, x2, y2 = box
            crop_img = cv2.imread(img_path)[int(y1) : int(y2), int(x1) : int(x2)]
            cls_ = infer_classify(crop_img, cls_model)
            labels.append(cls_)

            result = (
                f"{img_name} {int(cls_)} {box_xywhn[0]} {box_xywhn[1]} "
                f"{box_xywhn[2]} {box_xywhn[3]} {score}"
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
