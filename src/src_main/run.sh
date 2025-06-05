cd /home/quangsang/Study/maiAnhEm/soict-hackathon-2024/src/src_main

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

python /home/quangsang/Study/maiAnhEm/soict-hackathon-2024/src/src_main/old.py \
    --input /home/quangsang/Study/data/0006_cut.mp4 \
    --output_video result_violation.mp4 \
    --output_data violations_data.csv \
    --detection_method combined \
    --stop_line 385,348,858,355 \
    --green_zone 121,710,378,361,876,373,1069,704 \
    --red_zone 853,338,772,134,494,131,343,337 \
    --traffic_light_bbox 717,34,802,84 \
    --min_frames_in_green 5 \
    --trajectory_interpolation 15 \
    --tolerance 20 \
    --save_preview_frames 10 \
    --min_pixel_percentage 0.02 \
    --debug 

python /home/quangsang/Study/maiAnhEm/soict-hackathon-2024/src/src_main/traffic_violation_detector.py \
    --input /home/quangsang/Study/PBL6-Traffice-Surveillance/backend/uploads/videos/38213424-cea2-4689-b4b1-37112c836572.mov \
    --output_video /home/quangsang/Study/PBL6-Traffice-Surveillance/backend/uploads/processed/38213424-cea2-4689-b4b1-37112c836572/result_violation.mp4 \
    --output_data /home/quangsang/Study/PBL6-Traffice-Surveillance/backend/uploads/processed/38213424-cea2-4689-b4b1-37112c836572/violations_data.csv \
    --detection_method combined \
    --traffic_light_bbox 848,23,882,104 \
    --green_zone 60,355,744,337,871,590,19,597 \
    --red_zone 70,327,761,310,669,85,269,102 \
    --stop_line 65,340,755,329 \
    --min_frames_in_green 5 \
    --trajectory_interpolation 15 \
    --tolerance 20 \
    --save_preview_frames 10 \
    --min_pixel_percentage 0.02 \
    --max_dimension 1280

--traffic_light_bbox 848,23,882,104 \
--green_zone 60,355,744,337,871,590,19,597 \
--red_zone 70,327,761,310,669,85,269,102 \
--stop_line 65,340,755,329 \

Adding traffic_light_bbox: [ 848, 23, 882, 104 ]
Adding green_zone: [
   60, 355, 744, 337,
  871, 590,  19, 597
]
Adding red_zone: [
   70, 327, 761, 310,
  669,  85, 269, 102
]
Adding stop_line: [ 65, 340, 755, 329 ]
