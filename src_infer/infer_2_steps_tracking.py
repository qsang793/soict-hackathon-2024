import argparse
import os
from dataclasses import dataclass

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from utils.yolo_utils import visualize_images


@dataclass
class Config:
    detect_conf_thres: float = 0.3
    detect_iou_thres: float = 0.8
    classify_conf_thres: float = 0.9


def infer_classify(source, model) -> int:
    classify_results = model.predict(
        source=source,
        conf=Config.classify_conf_thres,
        verbose=False,
    )[0]

    cls_ = classify_results.probs.top1

    cls_fix = {
        0: 1,
        1: 2,
        2: 0,
        3: 3,
    }

    return cls_fix[cls_]


def infer_detect(source, model):
    detections = model.track(
        source=source,
        conf=Config.detect_conf_thres,
        iou=Config.detect_iou_thres,
        verbose=False,
        persist=True,
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
    det_model_path = "SOICT2024-VEHICLE-DETECTION-1CLS/run_1/weights/best.pt"
    det_model = YOLO(det_model_path)

    cls_model_path = "SOICT2024-VEHICLE-CLASSIFICATION/run_0/weights/best.pt"
    cls_model = YOLO(cls_model_path, task="classify")

    results = {}
    track_results = {}

    list_img = os.listdir(img_dir)
    list_img.sort()
    for img_name in tqdm(list_img, desc="Inferencing"):
        results[img_name] = {}

        img_path = os.path.join(img_dir, img_name)

        detections = infer_detect(img_path, det_model)

        if detections.boxes.id is None:
            print(f"No detections found in {img_name}")
            breakpoint()

        boxes = detections.boxes.xyxy.cpu().numpy()
        boxes_xywhn = detections.boxes.xywhn.cpu().numpy()
        scores = detections.boxes.conf.cpu().tolist()
        ids = detections.boxes.id.cpu().tolist()

        for box, id, box_xywhn, score in zip(boxes, ids, boxes_xywhn, scores):
            x1, y1, x2, y2 = box
            crop_img = cv2.imread(img_path)[int(y1) : int(y2), int(x1) : int(x2)]
            cls_ = infer_classify(crop_img, cls_model)

            if id not in track_results:
                track_results[id] = [0, 0, 0, 0]
            track_results[id][cls_] += 1

            results[img_name][id] = {
                "box": box_xywhn,
                "score": score,
            }

    ## Refine track results
    refine_track_results = {}
    for id, counts in track_results.items():
        cls_ = counts.index(max(counts))
        refine_track_results[id] = cls_

    txt_results = []
    for img_name in results:
        boxes = []
        labels = []
        for id, result in results[img_name].items():
            box = result["box"]
            score = result["score"]
            cls_ = refine_track_results[id]
            txt_results.append(
                f"{img_name} {cls_} {box[0]} {box[1]} {box[2]} {box[3]} {score}"
            )

            boxes.append(box)
            labels.append(cls_)

        if args.vis:
            img_path = os.path.join(img_dir, img_name)
            image = cv2.imread(img_path)

            breakpoint()
            visualized_img = visualize_images(image, boxes, labels)
            save_path = os.path.join(visualized_dir, img_name)
            cv2.imwrite(save_path, visualized_img)

    # Save results to file
    with open(output_path, "w") as f:
        f.write("\n".join(txt_results))
