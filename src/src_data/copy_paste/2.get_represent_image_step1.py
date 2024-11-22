import os
import shutil
import argparse

def organize_images_by_class(images_dir, labels_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in os.listdir(images_dir):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            parts = image_name.split('_')
            if len(parts) < 2:
                print(f"Filename {image_name} does not conform to expected format. Skipping.")
                continue
            class_part = parts[-1]
            class_id = os.path.splitext(class_part)[0].replace('class_', '')

            class_name = f'class_{class_id}'

            class_dir = os.path.join(output_dir, class_name)
            images_class_dir = os.path.join(class_dir, 'images')
            labels_class_dir = os.path.join(class_dir, 'labels')

            os.makedirs(images_class_dir, exist_ok=True)
            os.makedirs(labels_class_dir, exist_ok=True)

            image_output_path = os.path.join(images_class_dir, image_name)
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_output_path = os.path.join(labels_class_dir, label_name)

            source_image_path = os.path.join(images_dir, image_name)
            source_label_path = os.path.join(labels_dir, label_name)

            if not os.path.isfile(source_label_path):
                print(f"Label file {label_name} does not exist for image {image_name}. Skipping.")
                continue

            shutil.copy(source_image_path, image_output_path)
            shutil.copy(source_label_path, label_output_path)

    print("Done organizing images and labels by class with separate folders for images and labels.")

def main():
    parser = argparse.ArgumentParser(description='Organize images and labels by class.')
    parser.add_argument('--images_dir', type=str, required=True, help='Path to the images directory')
    parser.add_argument('--labels_dir', type=str, required=True, help='Path to the labels directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')

    args = parser.parse_args()

    organize_images_by_class(args.images_dir, args.labels_dir, args.output_dir)

if __name__ == '__main__':
    main()