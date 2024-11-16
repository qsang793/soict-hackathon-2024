import cv2

from utils.const import CLASSES_MAP, COLORS_MAP, REMAP_NIGHT_CLASSES


def convert_yolo_to_xyxy(bbox, img_shape):
    image_h, image_w = img_shape[:2]
    xc, yc, w, h = bbox
    x1 = int((xc - w / 2) * image_w)
    y1 = int((yc - h / 2) * image_h)
    x2 = int((xc + w / 2) * image_w)
    y2 = int((yc + h / 2) * image_h)
    return x1, y1, x2, y2


def visualize_images(image, boxes, labels):
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = COLORS_MAP[label]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image


def read_yolo_txt(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    boxes = []
    labels = []
    for line in lines:
        label, x, y, w, h = map(float, line.strip().split())
        boxes.append([x, y, w, h])
        if label in CLASSES_MAP:
            labels.append(int(label))
        else:
            labels.append(REMAP_NIGHT_CLASSES[int(label)])

    return boxes, labels
