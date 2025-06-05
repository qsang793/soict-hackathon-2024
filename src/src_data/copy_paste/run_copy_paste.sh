## Step 1 - Crop vehicles
python3 src/src_data/copy_paste/1_crop_vehicles.py --data_root demo_data --save_root __cropped

## Step 2 - Organize vehicles
python3 src/src_data/copy_paste/2_organize_vehicles.py --data_root __cropped --save_root __organized

## Step 3 - Get presented vehicles
python3 src/src_data/copy_paste/3_get_represent_image.py --data_root __organized/class_1 --save_root __presented_class_1
python3 src/src_data/copy_paste/3_get_represent_image.py --data_root __organized/class_2 --save_root __presented_class_2
python3 src/src_data/copy_paste/3_get_represent_image.py --data_root __organized/class_3 --save_root __presented_class_3

## Step 4 - Paste vehicles
python3 src/src_data/copy_paste/4_paste_vehicles.py --data_root demo_data --save_root __copy_paste --class_dirs __presented_class_1 __presented_class_2 __presented_class_3