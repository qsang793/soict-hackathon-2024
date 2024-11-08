import cv2


COLORS_MAP = {
    0: (0, 0, 255),  # red
    1: (0, 255, 0),  # green
    2: (255, 0, 0),  # blue
    3: (0, 255, 255),  # yellow
}

CLASSES_MAP = {
    0: "motorbike",
    1: "car",
    2: "coach",
    3: "truck",
}


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
        cv2.putText(
            image,
            CLASSES_MAP[label],
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

    return image


def read_yolo_txt(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    boxes = []
    labels = []
    for line in lines:
        label, x, y, w, h = map(float, line.strip().split())
        boxes.append([x, y, w, h])
        labels.append(int(label))

    return boxes, labels
