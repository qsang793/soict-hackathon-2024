"""Train 4 classes vehicle detection model"""

import torch
from ultralytics import YOLO

import wandb
from wandb.integration.ultralytics import add_wandb_callback


## Config
project = "SOICT2024-VEHICLE-DETECTION"
run_name = "run_hack"


pretrain_weight = "/home/manhckv/manhckv/soict/weights/vehicle/epoch140.pt"
data_yaml = "/home/manhckv/manhckv/soict/src/src_train/data-detection/data_train.yaml"

epochs = 200
batch_size = 32

patience = 500
cos_lr = False

device = [0, 1]
workers = 8

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
    patience=patience,
    cos_lr=cos_lr,
    imgsz=640,
    batch=batch_size,
    device=device,
    workers=workers,
    project=project,
    name=run_name,
    resume=resume,
    plots=plots,
)

wandb.finish()
