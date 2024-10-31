from ultralytics import YOLO


## Config
weight_path = ""
data_yaml = "/home/manhckv/manhckv/soict/src/data-yaml/data_train.yaml"

device = [0]
batch_size = 4
conf = 0.001
iou = 0.6

### Init model
model = YOLO(weight_path)

### Validate
metrics = model.val(device=device, batch=batch_size, data=data_yaml, conf=conf, iou=iou)
