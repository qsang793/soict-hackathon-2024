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

python /home/quangsang/Study/maiAnhEm/soict-hackathon-2024/src/src_main/traffic_violation_detector.py \
    --input /home/quangsang/Study/Traffic-Monitoring/models/data_test/DJI_20250520051359_0005_D.MP4 \
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