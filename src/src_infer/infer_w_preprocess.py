import argparse
import os

import cv2
import numpy as np
import torch
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor
from basicsr.utils import tensor2img
from basicsr.utils.options import parse
from tqdm import tqdm
from ultralytics import YOLO

from src.LoLi_IEA.LoLi_IEA import LoLi_IEA
from src.utils.yolo_utils import visualize_images


def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    parser.add_argument("--img_dir", type=str, default="public_test")
    parser.add_argument(
        "--vehicle_model_path", type=str, default="weights/vehicle/epoch40.pt"
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


def deblur(nafnet, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = _img2tensor(img, bgr2rgb=False, float32=True)

    nafnet.feed_data(data={"lq": img.unsqueeze(dim=0)})

    if nafnet.opt["val"].get("grids", False):
        nafnet.grids()

    nafnet.test()

    if nafnet.opt["val"].get("grids", False):
        nafnet.grids_inverse()

    visuals = nafnet.get_current_visuals()
    sr_img = tensor2img([visuals["result"]])
    return sr_img


def infer_classify(model, source) -> int:
    classify_results = model.predict(
        source=source,
        verbose=False,
    )[0]
    cls_ = classify_results.probs.top1
    return cls_


def infer_detect(model, source, conf=0.01, iou=0.7):
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

    opt_path = "weights/NAFNNet/NAFNet-width64.yml"
    opt = parse(opt_path, is_train=False)
    opt["dist"] = False
    opt["num_gpus"] = 1 if device.type == "cuda" else 0
    NAFNet = create_model(opt)

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

        if day_night_cls == 0:
            image = deblur(NAFNet, image)
        else:
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
            visualized_img = visualize_images(image, boxes, labels)
            save_path = os.path.join(visualized_dir, img_name)
            cv2.imwrite(save_path, visualized_img)

    ## Save results to file
    with open(output_path, "w") as f:
        for result in results:
            f.write(result + "\n")
