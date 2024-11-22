import os
import cv2
import numpy as np
import random
import argparse

def load_labels(label_path):
    """
    Load bounding boxes from the label file.
    """
    if not os.path.isfile(label_path):
        return []
    
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = abs(float(parts[3]))  
                height = abs(float(parts[4]))  
                labels.append((class_id, x_center, y_center, width, height))
            except ValueError:
                continue
    return labels

def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    """
    bbox1 = np.array(bbox1)
    bbox2 = np.array(bbox2)

    x_min = np.maximum(bbox1[0], bbox2[0])
    y_min = np.maximum(bbox1[1], bbox2[1])
    x_max = np.minimum(bbox1[2], bbox2[2])
    y_max = np.minimum(bbox1[3], bbox2[3])
    
    inter_width = np.maximum(0, x_max - x_min)
    inter_height = np.maximum(0, y_max - y_min)

    if inter_width <= 0 or inter_height <= 0:
        return 0.0

    inter_area = inter_width * inter_height

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def paste_image(base_image, paste_img, paste_box):
    """
    Paste an image onto the base image at the specified bounding box using NumPy.
    """
    x_min, y_min, x_max, y_max = map(int, paste_box)
    paste_width = x_max - x_min
    paste_height = y_max - y_min

    if paste_width <= 0 or paste_height <= 0:
        print("Invalid paste dimensions. Skipping this image.")
        return base_image

    paste_img_resized = cv2.resize(paste_img, (paste_width, paste_height))
    base_height, base_width, _ = base_image.shape
    x_min_clipped, y_min_clipped = np.clip([x_min, y_min], [0, 0], [base_width, base_height])
    x_max_clipped, y_max_clipped = np.clip([x_max, y_max], [0, 0], [base_width, base_height])
    paste_width_clipped = x_max_clipped - x_min_clipped
    paste_height_clipped = y_max_clipped - y_min_clipped

    if paste_width_clipped <= 0 or paste_height_clipped <= 0:
        print("After clipping, invalid paste dimensions. Skipping this image.")
        return base_image

    if (paste_width_clipped != paste_width) or (paste_height_clipped != paste_height):
        paste_img_resized = cv2.resize(paste_img, (paste_width_clipped, paste_height_clipped))

    base_image[y_min_clipped:y_max_clipped, x_min_clipped:x_max_clipped] = paste_img_resized

    return base_image

def normalize_label(label, image_width, image_height):
    """
    Normalize bounding box coordinates to the [0, 1] range using NumPy.
    """
    class_id, x_center, y_center, width, height = label
    coordinates = np.array([x_center, y_center, width, height], dtype=np.float32)

    if np.all(coordinates[2:] <= 1):
        return (class_id, *coordinates)

    normalized_coords = np.empty_like(coordinates)
    normalized_coords[0] = coordinates[0] / image_width   # x_center_norm
    normalized_coords[1] = coordinates[1] / image_height  # y_center_norm
    normalized_coords[2] = coordinates[2] / image_width   # width_norm
    normalized_coords[3] = coordinates[3] / image_height  # height_norm
    normalized_coords = np.clip(normalized_coords, 0, 1)

    return (class_id, *normalized_coords)

def augment_image(base_image_path, base_label_path, class_dirs, output_image_path, output_label_path, iou_threshold=0.1, max_attempts=20):
    """
    Augment the base image by pasting images from different classes.
    """
    base_image = cv2.imread(base_image_path)
    if base_image is None:
        print(f"Cannot read base image: {base_image_path}")
        return
    height, width, _ = base_image.shape

    base_labels = load_labels(base_label_path)
    normalized_base_labels = []
    for label in base_labels:
        normalized_label = normalize_label(label, width, height)
        normalized_base_labels.append(normalized_label)

    augmented_image = base_image.copy()
    new_labels = normalized_base_labels.copy()

    base_boxes = []
    for label in normalized_base_labels:
        bbox = [
            (label[1] - label[3] / 2) * width,  # x_min
            (label[2] - label[4] / 2) * height, # y_min
            (label[1] + label[3] / 2) * width,  # x_max
            (label[2] + label[4] / 2) * height  # y_max
        ]
        base_boxes.append(bbox)

    for class_dir in class_dirs:

        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(image_files) < 2:
            print(f"Not enough images in {class_dir}. Skipping this class.")
            continue

        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            selected_images = random.sample(image_files, 2)
            selected_boxes = []
            overlap_found = False

            for image_name in selected_images:
                label_name = os.path.splitext(image_name)[0] + '.txt'
                label_path = os.path.join(class_dir, label_name)
                labels = load_labels(label_path)
                if not labels:
                    print(f"No labels found for image: {image_name}. Skipping these images.")
                    overlap_found = True
                    break
                label = labels[0]
                bbox = [
                    label[1],  # x_center
                    label[2], # y_center
                    label[3],  # width
                    label[4]  # height
                ]
                selected_boxes.append(bbox)

            if overlap_found or len(selected_boxes) != 2:
                print(f"Attempt {attempts}: Invalid selection. Retrying with different images.")
                continue

            selected_absolute_bboxes = []
            for bbox in selected_boxes:
                x_center, y_center, bbox_width, bbox_height = bbox
                x_min = x_center - bbox_width / 2
                y_min = y_center - bbox_height / 2
                x_max = x_center + bbox_width / 2
                y_max = y_center + bbox_height / 2
                selected_absolute_bboxes.append([x_min, y_min, x_max, y_max])

            iou = calculate_iou(selected_absolute_bboxes[0], selected_absolute_bboxes[1])
            if iou > iou_threshold:
                print(f"Attempt {attempts}: IoU between selected images is {iou:.2f} > {iou_threshold}. Retrying.")
                continue

            for box in selected_absolute_bboxes:
                for base_box in base_boxes:
                    current_iou = calculate_iou(box, base_box)
                    if current_iou > iou_threshold:
                        print(f"Attempt {attempts}: Selected box overlaps with base box (IoU={current_iou:.2f}). Retrying.")
                        overlap_found = True
                        break
                if overlap_found:
                    break

            if overlap_found:
                continue

            for image_name, box in zip(selected_images, selected_absolute_bboxes):
                paste_img_path = os.path.join(class_dir, image_name)
                paste_img = cv2.imread(paste_img_path)
                if paste_img is None:
                    print(f"Cannot read image: {paste_img_path}. Skipping this image.")
                    continue
                augmented_image = paste_image(augmented_image, paste_img, box)
                
                class_id = labels[0][0]
                x_center_new = ((box[0] + box[2]) / 2) / width
                y_center_new = ((box[1] + box[3]) / 2) / height
                bbox_width_new = (box[2] - box[0]) / width
                bbox_height_new = (box[3] - box[1]) / height

                x_center_new = min(max(x_center_new, 0), 1)
                y_center_new = min(max(y_center_new, 0), 1)
                bbox_width_new = min(max(bbox_width_new, 0), 1)
                bbox_height_new = min(max(bbox_height_new, 0), 1)
                new_label = (int(class_id), x_center_new, y_center_new, bbox_width_new, bbox_height_new)
                new_labels.append(new_label)
                base_boxes.append(box)

                print(f"Pasted {image_name} onto base image.")

            print(f"Attempt {attempts}: Successfully augmented with images {selected_images} from {class_dir}.")
            break
        else:
            print(f"Reached maximum attempts for class {class_dir}. Skipping augmentation for this class.")

    cv2.imwrite(output_image_path, augmented_image)
    print(f"Saved augmented image to {output_image_path}")

    with open(output_label_path, 'w') as f:
        for label in new_labels:
            class_id, x_center, y_center, width, height = label
            x_center = min(max(float(x_center), 0), 1)
            y_center = min(max(float(y_center), 0), 1)
            width = min(max(float(width), 0), 1)
            height = min(max(float(height), 0), 1)
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    print(f"Saved updated labels to {output_label_path}")

def main():
    parser = argparse.ArgumentParser(description='Data augmentation script.')
    parser.add_argument('--base_images_dir', type=str, required=True, help='Path to the base images directory')
    parser.add_argument('--base_labels_dir', type=str, required=True, help='Path to the base labels directory')
    parser.add_argument('--class_dirs', nargs='+', required=True, help='List of class directories containing images and labels')
    parser.add_argument('--output_images_dir', type=str, required=True, help='Path to the output images directory')
    parser.add_argument('--output_labels_dir', type=str, required=True, help='Path to the output labels directory')
    parser.add_argument('--iou_threshold', type=float, default=0.05, help='IoU threshold to determine overlap')
    parser.add_argument('--max_attempts', type=int, default=50, help='Maximum attempts to find suitable images')

    args = parser.parse_args()

    os.makedirs(args.output_images_dir, exist_ok=True)
    os.makedirs(args.output_labels_dir, exist_ok=True)

    for image_name in os.listdir(args.base_images_dir):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue  # Skip non-image files

        base_image_path = os.path.join(args.base_images_dir, image_name)
        base_label_path = os.path.join(args.base_labels_dir, os.path.splitext(image_name)[0] + '.txt')
        output_image_path = os.path.join(args.output_images_dir, image_name)
        output_label_path = os.path.join(args.output_labels_dir, os.path.splitext(image_name)[0] + '.txt')

        print(f"Processing base image: {image_name}")
        augment_image(
            base_image_path,
            base_label_path,
            args.class_dirs,
            output_image_path,
            output_label_path,
            iou_threshold=args.iou_threshold,
            max_attempts=args.max_attempts
        )
        print("-" * 50)

    print("Data augmentation completed.")

if __name__ == '__main__':
    main()