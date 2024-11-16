"""Merge all labels to just 1 class (0) for all"""

import os

from tqdm import tqdm


if __name__ == "__main__":
    label_dir = "/home/manhckv/manhckv/soict/data/original/valid/labels"

    save_dir = "/home/manhckv/manhckv/soict/data/1_class/valid/labels"
    os.makedirs(save_dir, exist_ok=True)

    for txt_file in tqdm(os.listdir(label_dir), desc="Merging"):
        if not txt_file.endswith(".txt"):
            print(f"Skip {txt_file}")
            continue

        with open(os.path.join(label_dir, txt_file), "r") as f:
            lines = f.readlines()

        # Make all labels to just 1 class
        one_cls_file = ["0" + line[1:] for line in lines]

        # Save to new file
        with open(os.path.join(save_dir, txt_file), "w") as f:
            f.write("".join(one_cls_file))
