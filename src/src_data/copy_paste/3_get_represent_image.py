import argparse
import os
import shutil

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Extract representative images using clustering."
    )

    parser.add_argument(
        "--data_root", type=str, help="Path to the data root with organized images"
    )
    parser.add_argument(
        "--save_root", type=str, required=True, help="Path to the output directory"
    )
    parser.add_argument(
        "--num_clusters", type=int, default=5, help="Number of clusters for KMeans"
    )

    args = parser.parse_args()

    data_root = args.data_root
    image_dir = os.path.join(data_root, "images")
    label_dir = os.path.join(data_root, "labels")

    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)

    # Initialize the feature extraction pipeline
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipeline(
        task="image-feature-extraction",
        model="google/vit-base-patch16-224-in21k",
        device=DEVICE,
        pool=True,
        use_fast=True,
    )

    # Extract features from all images
    feature_list = []
    file_list = os.listdir(image_dir)
    for image_name in tqdm(file_list, desc="Extracting features"):
        image_path = os.path.join(image_dir, image_name)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_name}: {e}")
            continue
        outputs = pipe([image])
        feature = outputs[0][0]
        feature_list.append(feature)

    # Convert feature list to numpy array
    features = np.array(feature_list)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(features)

    # Find the closest image to each cluster centroid
    centroid_indices = []
    for centroid in kmeans.cluster_centers_:
        distances = np.linalg.norm(features - centroid, axis=1)
        closest_index = np.argmin(distances)
        centroid_indices.append(closest_index)

    # Save the representative images and labels
    for cluster, index in enumerate(centroid_indices):
        image_name = file_list[index]
        shutil.copy(
            os.path.join(image_dir, image_name),
            os.path.join(save_root, image_name),
        )

        # Copy the corresponding label file
        label_name = os.path.splitext(image_name)[0] + ".txt"
        shutil.copy(
            os.path.join(label_dir, label_name),
            os.path.join(save_root, label_name),
        )

    print("Done saving centroid images and labels.")


if __name__ == "__main__":
    main()
