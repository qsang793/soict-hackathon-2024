import os
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


pt_path = "/home/manhckv/vm7608/soict/weights/vehicle/best_private.pt"

task = "detect"

batch_size = 1

img_dir = "/home/manhckv/vm7608/soict/__data/test/all/images"

num_images = len(os.listdir(img_dir))
all_images = []
for img_name in tqdm(os.listdir(img_dir), desc="Read image"):
    image = cv2.imread(os.path.join(img_dir, img_name))
    all_images.append(image)

## Dummy frame
dummy_frame = np.zeros((640, 480, 3), dtype=np.uint8)
## Benchmark

# ## Load pytorch model
# pt_model = YOLO(pt_path, task=task)
# pt_model.predict(dummy_frame, verbose=False)
# start_time = time.perf_counter()
# for i in tqdm(range(0, len(all_images), batch_size), desc="Inferencing Pytorch"):
#     batch_images = all_images[i : i + batch_size]
#     results = pt_model.predict(batch_images)
# pt_time = time.perf_counter() - start_time
# del pt_model
# torch.cuda.empty_cache()


## Load the exported TensorRT model
engine_path = pt_path.replace(".pt", ".engine")
trt_model = YOLO(engine_path, task=task)
trt_model.predict(dummy_frame, device="cuda", verbose=False)
start_time = time.perf_counter()
for i in tqdm(range(0, len(all_images), batch_size), desc="Inferencing TensorRT"):
    batch_images = all_images[i : i + batch_size]
    results = trt_model.predict(batch_images)
trt_time = time.perf_counter() - start_time
del trt_model
torch.cuda.empty_cache()


# print(f"Pytorch inference time: {pt_time:.2f}s - FPS: {num_images / pt_time:.2f}")
print(f"TensorRT inference time: {trt_time:.2f}s - FPS: {num_images / trt_time:.2f}")
