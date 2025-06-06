#!/bin/bash

# Script to run the Integrated Traffic Violation Detection System
# This uses your original traffic_violation_detector.py with added license plate extraction

echo "🚀 Starting Integrated Traffic Violation Detection with License Plate Extraction"

# Change to the correct directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $(pwd)"

# Set environment variables
export GEMINI_API_KEY="AIzaSyDSF1qp4vK5AkAIKwO1_3kkaolkLxRomVU"  # Replace with your actual API key

# Default parameters (same as your original system)
VIDEO_INPUT="/home/quangsang/Study/data/0006_cut.mp4"
OUTPUT_VIDEO="output_integrated.mp4"
OUTPUT_DATA="violations_with_plates.csv"

# Check if video input file exists
if [ ! -f "$VIDEO_INPUT" ]; then
    echo "❌ Error: Video file not found: $VIDEO_INPUT"
    echo "Current working directory: $(pwd)"
    echo "Please check the video path."
    exit 1
fi

echo "📹 Input video: $VIDEO_INPUT"
echo "🎬 Output video: $OUTPUT_VIDEO"
echo "📊 Output data: $OUTPUT_DATA"

# Option 1: Test imports first
echo "🧪 Testing imports..."
python test_imports.py

if [ $? -ne 0 ]; then
    echo "❌ Import test failed! Please fix imports before running."
    exit 1
fi

echo "✅ Imports successful! Running integrated system..."

# Option 2: Run the simple Python script (recommended)
python run_simple.py

# Check if processing was successful
if [ $? -eq 0 ]; then
    echo "✅ Processing completed successfully!"
    echo ""
    echo "📁 Generated files:"
    echo "   🎬 Video with detections: $OUTPUT_VIDEO"
    echo "   📊 CSV with violations: $OUTPUT_DATA"
    echo "   📸 Cropped violation images: violation_images/"
    echo "   📄 Comprehensive report: violations_report.json"
    echo ""
    echo "🔍 License plate extraction results:"
    if [ -f "violations_report.json" ]; then
        python -c "
import json
try:
    with open('violations_report.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'   Total violations: {data[\"total_violations\"]}')
    plates_found = sum(1 for v in data['violations'] if v['license_plate'])
    print(f'   License plates found: {plates_found}')
    print(f'   Success rate: {plates_found/data[\"total_violations\"]*100:.1f}%' if data['total_violations'] > 0 else '   Success rate: N/A')
except:
    print('   Report file not generated or invalid')
"
    fi
else
    echo "❌ Processing failed!"
    exit 1
fi 