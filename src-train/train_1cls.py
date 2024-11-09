"""Train 1 class vehicle detection model"""

import torch
from ultralytics import YOLO

import wandb
from wandb.integration.ultralytics import add_wandb_callback


## Config
project = "SOICT2024-VEHICLE-DETECTION-1CLS"
run_name = "run_1"


pretrain_weight = "/home/manhckv/manhckv/soict/__weights/yolov9/yolov9e.pt"
data_yaml = "/home/manhckv/manhckv/soict/src-train/data-1cls/data_train_1cls.yaml"

epochs = 500
batch_size = 32

device = [0, 1]
workers = 8
cos_lr = True

resume = False
plots = False

## Init model
model = YOLO(pretrain_weight)
add_wandb_callback(model)

## Train
torch.cuda.empty_cache()
results = model.train(
    data=data_yaml,
    epochs=epochs,
    imgsz=640,
    batch=batch_size,
    device=device,
    workers=workers,
    cos_lr=cos_lr,
    project=project,
    name=run_name,
    resume=resume,
    plots=plots,
)

wandb.finish()
