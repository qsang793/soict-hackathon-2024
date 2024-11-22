import os
import shutil
import argparse
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser(description='Extract representative images using clustering.')
    parser.add_argument('--images_root', type=str, required=True, help='Path to the input images directory')
    parser.add_argument('--labels_root', type=str, required=True, help='Path to the input labels directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--number_clusters', type=int, default=200, help='Number of clusters for KMeans')

    args = parser.parse_args()

    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Initialize the feature extraction pipeline
    pipe = pipeline(
        task="image-feature-extraction",
        model="google/vit-base-patch16-224-in21k",
        device=DEVICE.index if DEVICE.type == 'cuda' else -1,
        pool=True,
        use_fast=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract features from all images
    feature_list = []
    file_list = os.listdir(args.images_root)
    for image_name in tqdm(file_list, desc="Extracting features"):
        image_path = os.path.join(args.images_root, image_name)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {image_name}: {e}")
            continue
        outputs = pipe([image])
        feature = outputs[0][0]
        feature_list.append(feature)

    # Convert feature list to numpy array
    features = np.array(feature_list)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=args.number_clusters, random_state=0).fit(features)

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
            os.path.join(args.images_root, image_name),
            os.path.join(args.output_dir, image_name),
        )

        # Copy the corresponding label file
        label_name = os.path.splitext(image_name)[0] + '.txt'
        shutil.copy(
            os.path.join(args.labels_root, label_name),
            os.path.join(args.output_dir, label_name),
        )

    print("Done saving centroid images and labels.")

if __name__ == '__main__':
    main()