import os
import argparse
from PIL import Image

def crop_images(images_dir, labels_dir, output_dir, output_labels_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    for image_name in os.listdir(images_dir):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_dir, image_name)
            label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')
            
            if not os.path.isfile(label_path):
                continue
            
            with Image.open(image_path) as img:
                width, height = img.size
                with open(label_path, 'r') as file:
                    for idx, line in enumerate(file):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)
                        
                        x_center *= width
                        y_center *= height
                        bbox_width *= width
                        bbox_height *= height
                        
                        x1 = int(x_center - bbox_width / 2)
                        y1 = int(y_center - bbox_height / 2)
                        x2 = int(x_center + bbox_width / 2)
                        y2 = int(y_center + bbox_height / 2)
                        
                        cropped_img = img.crop((x1, y1, x2, y2))
                        output_image_name = f"{os.path.splitext(image_name)[0]}_crop_{idx}_class_{int(class_id)}.jpg"
                        cropped_img.save(os.path.join(output_dir, output_image_name))
                        
                        # Save label file for the cropped image
                        output_label_name = f"{os.path.splitext(image_name)[0]}_crop_{idx}_class_{int(class_id)}.txt"
                        with open(os.path.join(output_labels_dir, output_label_name), 'w') as label_file:
                            label_file.write(f"{int(class_id)} {x_center} {y_center} {bbox_width} {bbox_height}\n")

def main():
    parser = argparse.ArgumentParser(description='Crop images based on labels and save them to output directories.')
    parser.add_argument('--images_dir', type=str, required=True, help='Path to the images directory')
    parser.add_argument('--labels_dir', type=str, required=True, help='Path to the labels directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output images directory')
    parser.add_argument('--output_labels_dir', type=str, required=True, help='Path to the output labels directory')

    args = parser.parse_args()

    crop_images(args.images_dir, args.labels_dir, args.output_dir, args.output_labels_dir)

if __name__ == '__main__':
    main()