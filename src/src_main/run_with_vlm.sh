#!/bin/bash

# Simple script to run Traffic Violation Detector with VLM addon
# This maintains the same performance as the original while adding license plate extraction

echo "🚀 Running Traffic Violation Detector with VLM License Plate Extraction"

# Set API key
export GEMINI_API_KEY="AIzaSyDSF1qp4vK5AkAIKwO1_3kkaolkLxRomVU"

# Input video
VIDEO_PATH="/home/quangsang/Study/data/0006_cut.mp4"

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
    --traffic_light_bbox 547,43,629,82 \
    --stop_line 391,363,916,375 \
    --green_zone 384,377,936,386,1150,713,164,716 \
    --red_zone 397,350,911,362,864,244,910,127,397,137,377,272 \
    --min_frames_in_green 5 \
    --trajectory_interpolation 15 \
    --tolerance 20 \
    --save_preview_frames 10 \
    --min_pixel_percentage 0.02 \
    --max_dimension 1920


echo "✅ Processing completed!"
echo "📁 Output files:"
echo "   🎬 Video: output_with_vlm.mp4"
echo "   📊 CSV: violations_with_vlm.csv"
echo "   📄 VLM Report: violations_with_plates.json"
echo "   📸 Violation images: violation_images/" 