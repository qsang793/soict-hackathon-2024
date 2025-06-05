from ultralytics import YOLO


## Config
weight_path = "/home/manhckv/vm7608/soict/weights/vehicle/best_private.pt"
data_yaml = "/home/manhckv/vm7608/soict/src/src_train/data-detection/data_test.yaml"

device = [0]
batch_size = 16
conf = 0.001
iou = 0.6

### Init model
model = YOLO(weight_path)

### Validate
metrics = model.val(device=device, batch=batch_size, data=data_yaml, conf=conf, iou=iou)
