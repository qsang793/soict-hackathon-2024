import torch
from ultralytics import YOLO


pt_path = "/home/manhckv/vm7608/soict/weights/vehicle/best_private.pt"

data_yaml = "/home/manhckv/vm7608/soict/src/src_train/data-detection/data_test.yaml"

task = "detect"

## Load pytorch model
pt_model = YOLO(pt_path)

## Load the exported TensorRT model
engine_path = pt_path.replace(".pt", ".engine")
trt_model = YOLO(engine_path, task=task)

device = [1]
batch_size = 4
conf = 0.001
iou = 0.6

## Validate Pytorch model --------------------------------
# print("Validate Pytorch model ----------------")
# pt_results = pt_model.val(
#     device=device, batch=batch_size, data=data_yaml, conf=conf, iou=iou
# )
# print("----------------------------------------")
# del pt_model
# torch.cuda.empty_cache()


## Validate TensorRT model -------------------------------
print("Validate TensorRT model ----------------")
trt_results = trt_model.val(
    data=data_yaml,
    batch=batch_size,
    conf=conf,
    iou=iou,
    device=device,
)
print("----------------------------------------")
del trt_model
torch.cuda.empty_cache()
