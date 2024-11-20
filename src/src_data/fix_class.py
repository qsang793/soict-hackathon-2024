import os

from tqdm import tqdm


REMAP_NIGHT_CLASSES = {
    4: 0,
    5: 1,
    6: 2,
    7: 3,
}


def fix_night_class(label_path):
    with open(label_path, "r") as f:
        lines = f.readlines()

    with open(label_path, "w") as f:
        for line in lines:
            class_id, x, y, w, h = line.strip().split()
            class_id = REMAP_NIGHT_CLASSES[int(class_id)]
            f.write(f"{class_id} {x} {y} {w} {h}\n")


if __name__ == "__main__":
    data_root = r"data\data_train\nighttime"
    images_dir = os.path.join(data_root, "images")
    labels_dir = os.path.join(data_root, "labels")

    for image_name in tqdm(os.listdir(images_dir)):
        label_path = os.path.join(labels_dir, image_name.rsplit(".", 1)[0] + ".txt")
        fix_night_class(label_path)
