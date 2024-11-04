import cv2
import random
import os

def load_labels(label_path, img_w, img_h):
    """Load bounding boxes from YOLO label file and convert to pixel coordinates."""
    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f]
    
    boxes = []
    for label in labels:
        x_center, y_center, width, height = map(float, label[1:])
        
        x_center, y_center = x_center * img_w, y_center * img_h
        box_w, box_h = width * img_w, height * img_h
        x_min, y_min = int(x_center - box_w / 2), int(y_center - box_h / 2)
        x_max, y_max = int(x_center + box_w / 2), int(y_center + box_h / 2)
        
        boxes.append([x_min, y_min, x_max, y_max])
    return boxes

def paste_sample_image(target_img, sample_img, existing_boxes):
    """
    Paste sample image onto target image at a position not overlapping with existing bounding boxes.
    """
    target_h, target_w, _ = target_img.shape
    sample_h, sample_w, _ = sample_img.shape
    
    max_attempts = 10
    for _ in range(max_attempts):
        x_offset = random.randint(0, target_w - sample_w)
        y_offset = random.randint(0, target_h - sample_h)
        new_box = [x_offset, y_offset, x_offset + sample_w, y_offset + sample_h]
        
        overlap = any(is_overlapping(new_box, box) for box in existing_boxes)
        if not overlap:
            target_img[y_offset:y_offset+sample_h, x_offset:x_offset+sample_w] = sample_img
            return new_box
    return None

def is_overlapping(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max <= x2_min or x1_min >= x2_max or y1_max <= y2_min or y1_min >= y2_max)

def augment_image(target_image_path, sample_image_path, label_path, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    target_img = cv2.imread(target_image_path)
    sample_img = cv2.imread(sample_image_path)

    img_h, img_w = target_img.shape[:2]

    existing_boxes = load_labels(label_path, img_w, img_h)
    
    target_img_copy = target_img.copy()
    
    new_box = paste_sample_image(target_img_copy, sample_img, existing_boxes)
    if new_box:
        filename = os.path.basename(target_image_path)
        save_path = os.path.join(save_directory, filename)
        cv2.imwrite(save_path, target_img_copy)
        print(f"Ảnh được lưu ở: {save_path}")
    else:
        print("Không có vị trí phù hợp để chèn ảnh mẫu.")

augment_image(
    target_image_path=r"D:\2_Hackathon\data_new\daytime\cam_01_00002.jpg",
    label_path=r"D:\2_Hackathon\data_new\daytime\cam_01_00002.txt",
    sample_image_path=r"D:\2_Hackathon\data_new\sample_object\truck.jpg",
    save_directory="augmentation_data"
)
