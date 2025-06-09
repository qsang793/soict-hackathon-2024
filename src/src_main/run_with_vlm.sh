#!/bin/bash

# Simple script to run Traffic Violation Detector with VLM addon
# This maintains the same performance as the original while adding license plate extraction

echo "🚀 Running Traffic Violation Detector with VLM License Plate Extraction"

# Set API key
export GEMINI_API_KEY="AIzaSyDSF1qp4vK5AkAIKwO1_3kkaolkLxRomVU"

# Input video
VIDEO_PATH="/home/quangsang/Study/data/0006_cut_1080.mov"

# Run the enhanced detector (same as original + VLM background processing)
python traffic_violation_detector_with_vlm.py \
    --input "$VIDEO_PATH" \
    --output_video "output_with_vlm.mp4" \
    --output_data "violations_with_vlm.csv" \
    --conf 0.65 \
    --iou 0.5 \
    --vehicle_model_path "/home/quangsang/Study/maiAnhEm/soict-hackathon-2024/weights/vehicle/epoch_best.pt" \
    --vis \
    --headless \
    --enable_vlm \
    --vlm_batch_size 3 \
    --debug \
    --detection_method combined \
    --traffic_light_bbox 825,70,954,122 \
    --stop_line 614,552,1364,568 \
    --green_zone 601,568,1387,584,1710,1066,289,1069 \
    --red_zone 619,539,1362,552,1349,221,581,214 \
    --min_frames_in_green 5 \
    --trajectory_interpolation 15 \
    --tolerance 20 \
    --save_preview_frames 10 \
    --min_pixel_percentage 0.02 \
    --max_dimension 1920

# green_zone
# : 
# (8) [601, 568, 1387, 584, 1710, 1066, 289, 1069]
# red_zone
# : 
# (8) [619, 539, 1362, 552, 1349, 221, 581, 214]
# stop_line
# : 
# (4) [614, 552, 1364, 568]
# traffic_light_bbox
# : 
# (4) [825, 70, 954, 122]

echo "✅ Processing completed!"
echo "📁 Output files:"
echo "   🎬 Video: output_with_vlm.mp4"
echo "   📊 CSV: violations_with_vlm.csv"
echo "   📄 VLM Report: violations_with_plates.json"
echo "   📸 Violation images: violation_images/" 