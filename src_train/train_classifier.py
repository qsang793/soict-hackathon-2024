"""Train 4 classes vehicle detection model"""

from ultralytics import YOLO

import wandb
from wandb.integration.ultralytics import add_wandb_callback


## Config
project = "SOICT2024-VEHICLE-CLASSIFICATION"
run_name = "run_0"


pretrain_weight = "/home/manhckv/manhckv/soict/__weights/yolov11-cls/yolo11x-cls.pt"
data_dir = "/home/manhckv/manhckv/soict/data/day_night_classify"

epochs = 200
batch_size = 32

device = [0, 1]
workers = 8

resume = False
plots = False

## Init model
model = YOLO(pretrain_weight)
add_wandb_callback(model)

## Train
results = model.train(
    data=data_dir,
    epochs=epochs,
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
