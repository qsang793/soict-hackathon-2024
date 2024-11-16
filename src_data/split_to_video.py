import os

import cv2


def parse_filename(image_dir):
    all_filename = os.listdir(image_dir)
    unique_prefixes = set()

    for img_name in all_filename:
        prefix = img_name.split("_")[:2]
        unique_prefixes.add("_".join(prefix))

    ## get list file name by prefix
    file_per_prefix = {}
    for prefix in unique_prefixes:
        file_per_prefix[prefix] = []
        for img_name in all_filename:
            if img_name.startswith(prefix):
                file_per_prefix[prefix].append(img_name)

    ## print number of images by prefix
    for prefix in unique_prefixes:
        print(f"{prefix}: {len(file_per_prefix[prefix])} images")

    return file_per_prefix


def images_to_video(images, save_path, height=720, width=1280, fps=3):
    video_writer = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for image in images:
        video_writer.write(image)
    video_writer.release()


if __name__ == "__main__":
    save_dir = r"D:\Project\SoICT2024\__temp"

    image_dir = r"D:\Project\SoICT2024\data\public_test"
    img_per_prefix = parse_filename(image_dir)

    for prefix, img_list in img_per_prefix.items():
        images = []
        for img_name in img_list:
            img = cv2.imread(os.path.join(image_dir, img_name))
            images.append(img)

        save_path = os.path.join(save_dir, f"{prefix}.mp4")
        images_to_video(images, save_path)
