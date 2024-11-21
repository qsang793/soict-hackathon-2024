"""Train 4 classes vehicle detection model"""

from ultralytics import YOLO

import wandb
from wandb.integration.ultralytics import add_wandb_callback


## Config
project = "SOICT2024-DAY-NIGHT-CLASSIFY"
run_name = "run_1"


pretrain_weight = "/home/manhckv/manhckv/soict/__pretrained_weights/yolo11x-cls.pt"
data_dir = "/home/manhckv/manhckv/soict/data/day_night_classify"

epochs = 100
batch_size = 32

device = [1]
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
