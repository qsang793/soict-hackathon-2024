from ultralytics import YOLO


pt_path = "/home/manhckv/vm7608/soict/weights/vehicle/best_private.pt"


model = YOLO(pt_path)

# Export the model to TensorRT format
engine_path = model.export(
    format="engine",
    imgsz=640,
    dynamic=True,
    half=True,
)
