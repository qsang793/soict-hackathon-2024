#!/bin/bash

# Simple script to run Traffic Violation Detector with VLM addon
# This maintains the same performance as the original while adding license plate extraction

echo "🚀 Running Traffic Violation Detector with VLM License Plate Extraction"

# Set API key
export GEMINI_API_KEY="AIzaSyDSF1qp4vK5AkAIKwO1_3kkaolkLxRomVU"

# Input video
VIDEO_PATH="/home/quangsang/Study/data/0006_cut.mp4"

# Run the enhanced detector (same as original + VLM background processing)
python traffic_violation_detector.py \
    --input "$VIDEO_PATH" \
    --output_video "output_with_vlm.mp4" \
    --output_data "violations_with_vlm.csv" \
    --conf 0.65 \
    --iou 0.5 \
    --headless \
    --enable_vlm \
    --vlm_batch_size 1 \
    --debug \
    --detection_method combined \
    --traffic_light_bbox 1655,141,1897,246 \
    --stop_line 1217,1101,2734,1130 \
    --green_zone 1182,1117,2779,1149,3424,2136,547,2126 \
    --red_zone 1202,1078,2744,1104,2668,406,1172,422 \
    --min_frames_in_green 5 \
    --trajectory_interpolation 15 \
    --tolerance 20 \
    --save_preview_frames 10 \
    --min_pixel_percentage 0.02 \
    --max_dimension 3840 \
    --use_tensorrt \
    --tensorrt_workspace 16 \
    --half_precision

echo "✅ Processing completed!"
echo "📁 Output files:"
echo "   🎬 Video: output_with_vlm.mp4"
echo "   📊 CSV: violations_with_vlm.csv"
echo "   📄 VLM Report: violations_with_plates.json"
echo "   📸 Violation images: violation_images/" 