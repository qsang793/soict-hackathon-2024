import argparse
import os
import time

import cv2
import numpy as np
import torch
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor
from basicsr.utils import tensor2img
from basicsr.utils.options import parse
from tqdm import tqdm


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.0
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)


def single_image_inference(model, img):
    model.feed_data(data={"lq": img.unsqueeze(dim=0)})

    if model.opt["val"].get("grids", False):
        model.grids()

    model.test()

    if model.opt["val"].get("grids", False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals["result"]])
    return sr_img


def parse_arguments():
    parser = argparse.ArgumentParser(description="Deblur with NAFNet")
    parser.add_argument(
        "--img_dir", type=str, help="Path to the folder with images to deblur"
    )
    parser.add_argument(
        "--save_dir", type=str, help="Path to folder to save the deblurred images"
    )
    return parser.parse_args()


if __name__ == "__main__":
    ## Parse arguments --------
    args = parse_arguments()

    ## Create model --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt_path = "weights/NAFNNet/NAFNet-width64.yml"
    opt = parse(opt_path, is_train=False)
    opt["dist"] = False
    opt["num_gpus"] = 1 if device.type == "cuda" else 0
    NAFNet = create_model(opt)

    ## Inference --------
    image_dir = args.img_dir

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    start_time = time.perf_counter()
    for image_name in tqdm(os.listdir(image_dir), desc="Inferencing"):
        image_path = os.path.join(image_dir, image_name)

        img_input = imread(image_path)
        inp = img2tensor(img_input)

        rs_img = single_image_inference(NAFNet, inp)

        save_path = os.path.join(save_dir, image_name)
        cv2.imwrite(save_path, rs_img)

    end_time = time.perf_counter()
    print(f"FPS: {len(os.listdir(image_dir)) / (end_time - start_time)}")
