"""Train 4 classes vehicle detection model"""

from ultralytics import YOLO

import wandb
from wandb.integration.ultralytics import add_wandb_callback


## Config
project = "SOICT2024-VEHICLE-DETECTION"
run_name = "run_2"


pretrain_weight = "/home/manhckv/manhckv/soict/__weights/yolov9/yolov9e.pt"
data_yaml = "/home/manhckv/manhckv/soict/src/data-yaml/data_train_full.yaml"

epochs = 400
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
    data=data_yaml,
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
