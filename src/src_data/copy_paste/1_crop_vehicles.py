import argparse
import os

from PIL import Image


def crop_images(image_dir, label_dir, save_image_dir, save_label_dir):
    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(
                label_dir, os.path.splitext(image_name)[0] + ".txt"
            )

            if not os.path.isfile(label_path):
                continue

            with Image.open(image_path) as img:
                width, height = img.size
                with open(label_path, "r") as file:
                    for idx, line in enumerate(file):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        class_id, x_center, y_center, bbox_width, bbox_height = map(
                            float, parts
                        )

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
                        cropped_img.save(
                            os.path.join(save_image_dir, output_image_name)
                        )

                        # Save label file for the cropped image
                        output_label_name = f"{os.path.splitext(image_name)[0]}_crop_{idx}_class_{int(class_id)}.txt"
                        with open(
                            os.path.join(save_label_dir, output_label_name), "w"
                        ) as label_file:
                            label_file.write(
                                f"{int(class_id)} {x_center} {y_center} {bbox_width} {bbox_height}\n"
                            )


def main():
    parser = argparse.ArgumentParser(
        description="Crop images based on labels and save them to output directories."
    )
    parser.add_argument(
        "--data_root", type=str, help="Path to the data root with images and labels"
    )
    parser.add_argument(
        "--save_root", type=str, help="Path to folder to save the cropped images"
    )

    args = parser.parse_args()

    data_root = args.data_root
    image_dir = os.path.join(data_root, "images")
    label_dir = os.path.join(data_root, "labels")

    save_root = args.save_root
    save_image_dir = os.path.join(save_root, "images")
    save_label_dir = os.path.join(save_root, "labels")

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    crop_images(image_dir, label_dir, save_image_dir, save_label_dir)


if __name__ == "__main__":
    main()
