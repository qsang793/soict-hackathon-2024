python src/src_infer/all_in.py \
  --input /home/quangsang/Study/Traffic-Monitoring/models/Test/RPReplay_Final1745372627.mov \
  --output_video result.mp4 \
  --max_dimension 640 \
  --save_preview_frames 30 \
  --headless

python src/src_infer/ultralytic.py \
  --input /home/quangsang/Study/Traffic-Monitoring/models/Test/RPReplay_Final1745372627.mov \
  --output_video result_ultralytic.mp4 \
  --max_dimension 640 \
  --save_preview_frames 30 \
  --headless

python src/src_infer/violation.py \
  --input /home/quangsang/Study/Traffic-Monitoring/models/Test/RPReplay_Final1745372627.mov \
  --output_video result_violation.mp4 \
  --max_dimension 640 \
  --save_preview_frames 30 \
  --headless \
  --violation_line 500,300,900,300 \
  --violation_direction any \
  --violation_classes all 

python src/src_infer/violation.py \
  --input /home/quangsang/Study/Traffic-Monitoring/models/Test/RPReplay_Final1745372627.mov \
  --output_video result_violation.mp4 \
  --max_dimension 640 \
  --save_preview_frames 30 \
  --headless \
  --detection_method combined --red_light \
  --stop_y 400 --tolerance 10

python src/src_infer/test.py \
  --input /home/quangsang/Study/Traffic-Monitoring/models/Test/RPReplay_Final1745372627.mov \
  --output_video result_violation.mp4 \
  --output_data violations_data.csv \
  --license_plate_model_path /home/quangsang/Study/Automatic-License-Plate-Recognition-using-YOLOv8/license_plate_detector.pt \
  --red_light \
  --detection_method combined \
  --stop_line 20,350,1000,350 \
  --green_zone 100,380,1000,380,1000,720,100,720 \
  --red_zone 100,0,1000,0,1000,320,100,320 \
  --min_frames_in_green 5 \
  --trajectory_interpolation 15 \
  --tolerance 20 \
  --save_preview_frames 10 \
  --use_tensorrt \
  --half_precision

python /home/quangsang/Study/maiAnhEm/soict-hackathon-2024/src/src_main/traffic_violation_detector.py \
  --traffic_light_bbox 858,32,883,99 \
  --input /home/quangsang/Study/Traffic-Monitoring/models/data_test/RPReplay_Final1745372627.mov \
  --output_video result_violation.mp4 \
  --output_data violations_data.csv \
  --detection_method combined \
  --stop_line 20,350,1000,350 \
  --green_zone 100,380,1000,380,1000,720,100,720 \
  --red_zone 100,0,1000,0,1000,320,100,320 \
  --min_frames_in_green 5 \
  --trajectory_interpolation 15 \
  --tolerance 20 \
  --save_preview_frames 10 \
  --use_tensorrt \
  --half_precision

python /home/quangsang/Study/maiAnhEm/soict-hackathon-2024/src/src_main/traffic_violation_detector.py \
    --input /home/quangsang/Study/Traffic-Monitoring/models/data_test/RPReplay_Final1745372627.mov \
    --traffic_light_bbox 858,32,883,99 \
    --output_video result_violation.mp4 \
    --output_data violations_data.csv \
    --detection_method combined \
    --stop_line 20,350,1000,350 \
    --green_zone 100,380,1000,380,1000,720,100,720 \
    --red_zone 100,0,1000,0,1000,320,100,320 \
    --min_frames_in_green 5 \
    --trajectory_interpolation 15 \
    --tolerance 20 \
    --save_preview_frames 10 \
    --use_hsv \
    --min_pixel_percentage 0.02 \
    --debug
    # --red_threshold_hsv "0,50,50,10,255,255,160,50,50,180,255,255" \
    # --yellow_threshold_hsv "15,50,50,35,255,255" \
    # --green_threshold_hsv "70,40,40,100,255,255" \
