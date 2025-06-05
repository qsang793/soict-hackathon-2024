1. Cài đặt
- Team đã có submit file docker có tên là "export_image.tar", để chạy ứng dụng, bạn cần phải import file này vào docker bằng cách:
```
docker load -i export_image.tar
```
- Sau khi import xong, sẽ có docker image có tên là "manhckv/test:latest"

2. Chạy ứng dụng
- Để chạy ứng dụng, bạn cần phải chạy lệnh sau:
```
docker run -it --runtime=nvidia --gpus all manhckv/test:latest
```

- Đầu tiên, cần chạy lệnh sau để cài đặt NAFNet:
```
bash scripts/install_NAFNet.sh
```

- Để chạy inference ra kết quả, bạn cần phải chạy lệnh sau:
```
python3 src/src_infer/infer_pipeline.py
```

- Kết quả sẽ được lưu vào file "src/predict.txt". Để copy kết quả ra khỏi docker, bạn cần phải chạy lệnh sau (chay lệnh này ở terminal khác, không phải trong docker):
```
docker cp <container_id>:/src/predict.txt .
```

- Trong đó container_id là id của container mà bạn vừa chạy ứng dụng (có thể xem bằng lệnh "docker ps -a"). Ví dụ container có id là "2399eae3124d", lệnh sẽ là:

```
docker cp 2399eae3124d:/src/predict.txt .
```

- Lúc này, kết quả sẽ được lưu vào file "predict.txt" ở thư mục hiện tại.

3. Augmentation data

- Ở trong container hiện tại, team đã có sẵn dữ liệu demo để chạy augmentation data. Dữ liệu demo nằm ở thư mục "demo_data". Để chạy augmentation data, bạn cần phải chạy các bước sau:

- Để chạy bước điều chỉnh độ sáng, bạn cần phải chạy lệnh sau. Sau khi chạy xong, kết quả sẽ được lưu vào thư mục "__adjust_exposure":
```
python3 src/src_data/adjust_exposure.py --data_root demo_data --save_root __adjust_exposure
```

- Để chạy bước convert ảnh sang grayscale, bạn cần phải chạy lệnh sau. Sau khi chạy xong, kết quả sẽ được lưu vào thư mục "__grayscale":
```
python3 src/src_data/convert_grayscale.py --data_root demo_data --save_root __grayscale
```

- Để chạy bước thêm motion blur, bạn cần phải chạy lệnh sau. Sau khi chạy xong, kết quả sẽ được lưu vào thư mục "__motion_blur":
```
python3 src/src_data/motion_blur.py --data_root demo_data --save_root __motion_blur
```

- Để chạy bước xóa xe máy trong hình ảnh, bạn cần phải chạy lệnh sau. Sau khi chạy xong, kết quả sẽ được lưu vào thư mục "__remove_motorbike":
```
python3 src/src_data/object_removal/remove_motorbike.py --data_root demo_data --save_root __remove_motorbike
```

- Để chạy bước copy và dán xe giữa các ảnh trong cùng 1 camera, bạn cần phải chạy lệnh sau. Sau khi chạy xong, kết quả sẽ được lưu vào thư mục "__copy_paste":
```
bash src/src_data/copy_paste/run_copy_paste.sh
```

4. Train model
- Để train model, bạn cần phải chạy docker với lệnh sau (cần có GPU):
```
docker run -it --runtime=nvidia --gpus all --shm-size=2G manhckv/test:latest
```

- Đầu tiên, cần chạy lệnh sau để tải dữ liệu training vehicle detection và day-night classification (cần có kết nối internet):
```
bash scripts/download_data.sh
```

- Sau đó, chạy lệnh sau để train model day-night classification:
```
python3 src/src_train/train_classifier.py
```

- Cuối cùng, chạy lệnh sau để train model vehicle detection:
```
python3 src/src_train/train_vehicle_detection.py
```

- Lưu ý: ở đây để phục vụ BTC test code training model, team đã config batchsize là 4 và epoch là 1. Để train model với batchsize và epoch lớn hơn, bạn cần phải chỉnh sửa trong file "src/src_train/train_vehicle_detection.py" và "src/src_train/train_classifier.py".