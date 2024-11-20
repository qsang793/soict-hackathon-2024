import argparse
import os

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from utils.yolo_utils import visualize_images


def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    parser.add_argument("--model_path", type=str, default="__project_weights/best.pt")
    parser.add_argument("--img_dir", type=str, default="data/public_test")
    parser.add_argument("--output_path", type=str, default="predict.txt")

    parser.add_argument(
        "--vis", action="store_true", help="Visualize the images with detections"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.vis:
        visualized_dir = "__visualized"
        os.makedirs(visualized_dir, exist_ok=True)

    img_dir = args.img_dir
    output_path = args.output_path

    # Load a model
    model_path = args.model_path
    model = YOLO(model_path)

    conf = 0.5
    iou = 0.5

    results = []
    for img_name in tqdm(os.listdir(img_dir), desc="Inferencing"):
        img_path = os.path.join(img_dir, img_name)

        detections = model.predict(
            img_path,
            conf=conf,
            iou=iou,
            verbose=False,
        )[0]

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
