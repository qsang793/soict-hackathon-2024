import os
import shutil
import argparse

def copy_files(file_list, src_folder, dest_folder):
    for image_file in file_list:
        image_path = os.path.join(src_folder, image_file)
        label_path = os.path.join(src_folder, os.path.splitext(image_file)[0] + '.txt')

        shutil.copy(image_path, os.path.join(dest_folder, 'images', image_file))
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(dest_folder, 'labels', os.path.splitext(image_file)[0] + '.txt'))

def main():
    parser = argparse.ArgumentParser(description='Copy images and labels for a specific camera.')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to the source images folder')
    parser.add_argument('--new_folder', type=str, required=True, help='Path to the destination folder')
    parser.add_argument('--camera_prefix', type=str, required=True, help='Camera prefix to filter images (e.g., cam_08_)')

    args = parser.parse_args()

    image_folder = args.image_folder
    new_folder = args.new_folder
    camera_prefix = args.camera_prefix

    os.makedirs(os.path.join(new_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(new_folder, 'labels'), exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') and f.startswith(camera_prefix)]

    copy_files(image_files, image_folder, new_folder)

if __name__ == '__main__':
    main()