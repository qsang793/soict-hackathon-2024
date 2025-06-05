"""
Example usage of the refactored ViolationDetector class

This example shows how to use the new organized violation detection system
with proper dimension handling and backend synchronization.
"""

import argparse
import cv2
from violation_detector import ViolationDetector, GeometryUtils


def create_sample_args():
    """Create sample arguments for testing"""
    class Args:
        def __init__(self):
            self.detection_method = "combined"
            self.green_zone = "100,300,500,300,500,400,100,400"  # Sample coordinates
            self.red_zone = "100,100,500,100,500,250,100,250"   # Sample coordinates
            self.stop_line = "100,250,500,250"                  # Sample stop line
            self.stop_y = None
            self.tolerance = 10
            self.violation_classes = "all"
            self.min_frames_in_green = 3
            self.trajectory_interpolation = 5
            self.red_light = False
            self.red_light_frames = "0,100,200,300"  # Sample red light phases
            self.debug = True
    
    return Args()


def example_video_processing():
    """Example of processing a video with violation detection"""
    
    # Create detector with sample arguments
    args = create_sample_args()
    detector = ViolationDetector(args)
    
    # Example video dimensions (simulating a real video)
    original_width, original_height = 1920, 1080
    processing_width, processing_height = 1280, 720
    
    # Set video dimensions for proper coordinate scaling
    detector.set_video_dimensions(
        original_width, original_height,
        processing_width, processing_height
    )
    
    # Set up default zones if needed
    detector.setup_default_zones(processing_width, processing_height)
    
    print("\n=== Starting violation detection simulation ===")
    
    # Simulate processing frames
    fps = 30
    sample_detections = [
        {
            'track_id': 1,
            'label': 0,  # Car
            'box': [200, 200, 250, 240],  # x1, y1, x2, y2
            'confidence': 0.8
        },
        {
            'track_id': 2, 
            'label': 1,  # Truck
            'box': [300, 180, 360, 220],
            'confidence': 0.9
        }
    ]
    
    # Process multiple frames
    for frame_count in range(100):
        # Simulate vehicle movement
        for detection in sample_detections:
            # Move vehicles down (simulating approach to stop line)
            detection['box'][1] += 2  # y1
            detection['box'][3] += 2  # y2
        
        # Process detections for violations
        new_violations = detector.process_detections(sample_detections, frame_count, fps)
        
        if new_violations > 0:
            print(f"Frame {frame_count}: {new_violations} new violations detected!")
    
    print(f"\n=== Processing complete ===")
    print(f"Total violations detected: {detector.violation_count}")
    print(f"Violations by vehicle: {list(detector.violations.keys())}")


def example_coordinate_scaling():
    """Example of coordinate scaling between different dimensions"""
    
    print("\n=== Coordinate Scaling Example ===")
    
    # Original video dimensions (from upload)
    original_width, original_height = 1920, 1080
    
    # Processing dimensions (backend automatically calculates)
    max_dimension = 1280
    if max(original_width, original_height) > max_dimension:
        scale_factor = max_dimension / max(original_width, original_height)
        processing_width = int(original_width * scale_factor)
        processing_height = int(original_height * scale_factor)
    else:
        processing_width, processing_height = original_width, original_height
        scale_factor = 1.0
    
    print(f"Original dimensions: {original_width}x{original_height}")
    print(f"Processing dimensions: {processing_width}x{processing_height}")
    print(f"Scale factor: {scale_factor:.3f}")
    
    # Example zone coordinates (from frontend, in processing dimensions)
    zone_coords_processing = [(100, 300), (500, 300), (500, 400), (100, 400)]
    
    # Convert to original dimensions (if needed for display)
    zone_coords_original = [(x / scale_factor, y / scale_factor) for x, y in zone_coords_processing]
    
    print(f"\nZone coordinates (processing): {zone_coords_processing}")
    print(f"Zone coordinates (original): {zone_coords_original}")
    
    print("\n✅ This ensures zones work correctly regardless of video size!")


def example_backend_integration():
    """Example of how the system integrates with backend"""
    
    print("\n=== Backend Integration Example ===")
    
    # Step 1: Video upload (backend gets metadata)
    video_metadata = {
        'originalWidth': 1920,
        'originalHeight': 1080, 
        'processingWidth': 1280,
        'processingHeight': 720,
        'scaleFactor': 0.667,
        'maxDimension': 1280
    }
    
    print("1. Video uploaded, metadata extracted:")
    print(f"   - Original: {video_metadata['originalWidth']}x{video_metadata['originalHeight']}")
    print(f"   - Processing: {video_metadata['processingWidth']}x{video_metadata['processingHeight']}")
    
    # Step 2: Frontend configures zones (automatically synced)
    print("\n2. Frontend zone configuration:")
    print("   - Zones are configured in processing dimensions")
    print("   - Frontend automatically syncs with backend metadata")
    
    zone_config = {
        'green_zone': [100, 300, 500, 300, 500, 400, 100, 400],
        'red_zone': [100, 100, 500, 100, 500, 250, 100, 250],
        'stop_line': [100, 250, 500, 250]
    }
    
    # Step 3: Backend runs inference with correct dimensions
    print("\n3. Backend inference:")
    print("   - Uses metadata.processingWidth for --max_dimension")
    print("   - Zone coordinates are already in correct format")
    print("   - No coordinate conversion needed!")
    
    print(f"\nCommand: python violation.py --max_dimension {video_metadata['processingWidth']} \\")
    print(f"         --green_zone {','.join(map(str, zone_config['green_zone']))} \\")
    print(f"         --red_zone {','.join(map(str, zone_config['red_zone']))} \\") 
    print(f"         --stop_line {','.join(map(str, zone_config['stop_line']))}")
    
    print("\n✅ Perfect synchronization between frontend and backend!")


if __name__ == "__main__":
    print("🚀 Refactored Traffic Violation Detection System")
    print("=" * 50)
    
    # Run examples
    example_coordinate_scaling()
    example_backend_integration()
    example_video_processing()
    
    print("\n" + "=" * 50)
    print("✅ All examples completed successfully!")
    print("\nKey benefits of the refactored system:")
    print("- 🎯 Organized, maintainable code")
    print("- 📐 Proper dimension handling")
    print("- 🔄 Backend synchronization")
    print("- 🧪 Easy testing and debugging")
    print("- 📈 Scalable for new features") 