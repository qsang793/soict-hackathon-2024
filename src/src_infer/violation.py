import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor
from basicsr.utils import tensor2img
from basicsr.utils.options import parse
from tqdm import tqdm
from ultralytics import YOLO

from LoLi_IEA.LoLi_IEA import LoLi_IEA
from utils.yolo_utils import visualize_images

def parse_arguments():
    parser = argparse.ArgumentParser(description="Traffic Violation Detection")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to video file or camera index (e.g. 0 for webcam)")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="Optional: Process images from directory instead of video")
    
    parser.add_argument("--conf", type=float, default=0.65)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--vehicle_model_path", type=str, default="weights/vehicle/epoch_best.pt")
    parser.add_argument("--daynight_model_path", type=str, default="weights/day_night/best.pt")
    parser.add_argument("--loli_iea_model_dir", type=str, default="weights/LoLi_IEA")

    parser.add_argument("--output_video", type=str, default="output.mp4")
    parser.add_argument("--output_data", type=str, default="tracking_data.csv")
    parser.add_argument("--vis", action="store_true", help="Visualize the detections in real-time")
    parser.add_argument("--headless", action="store_true", 
                        help="Run in headless mode without any display windows")
    parser.add_argument("--skip_frames", type=int, default=0, 
                        help="Process every Nth frame only (0=process all frames)")
    parser.add_argument("--process_resolution", type=str, default=None,
                        help="Process at this resolution, format: WIDTHxHEIGHT")
    parser.add_argument("--memory_efficient", action="store_true", 
                        help="Use CPU for enhancement models to save GPU memory")
    parser.add_argument("--disable_enhancement", action="store_true",
                        help="Disable image enhancement to save memory")
    parser.add_argument("--max_dimension", type=int, default=1280,
                        help="Maximum dimension for processing (larger images will be resized)")
    parser.add_argument("--trajectory_length", type=int, default=30,
                        help="Length of trajectory history to display")
    parser.add_argument("--save_preview_frames", type=int, default=0,
                        help="Save preview frames at intervals (0 to disable)")
    
    # Tracking specific parameters
    parser.add_argument("--tracker", type=str, default="bytetrack", 
                        choices=["bytetrack", "botsort"],
                        help="Tracker algorithm to use")
    parser.add_argument("--tracker_config", type=str, default=None,
                        help="Optional tracker configuration file")
    
    # Stop line parameters
    parser.add_argument("--stop_line", type=str, default=None,
                        help="Format: x1,y1,x2,y2 - Coordinates for the stop line")
    parser.add_argument("--stop_y", type=int, default=None,
                        help="Y-coordinate of horizontal stop line (used if stop_line is None)")
    parser.add_argument("--tolerance", type=int, default=10,
                        help="Tolerance in pixels for the stop line detection")
    
    # Zone parameters
    parser.add_argument("--red_zone", type=str, default=None,
                        help="Format: x1,y1,x2,y2,x3,y3,... - Coordinates for the red (violation) zone")
    parser.add_argument("--green_zone", type=str, default=None,
                        help="Format: x1,y1,x2,y2,x3,y3,... - Coordinates for the green (legal) zone")
    
    # Red light parameters
    parser.add_argument("--red_light", action="store_true",
                        help="Simulate red light phase (always on)")
    parser.add_argument("--red_light_frames", type=str, default=None,
                        help="Format: start1,end1,start2,end2,... - Frame ranges for red light phases")
    
    # Violation parameters
    parser.add_argument("--violation_cooldown", type=int, default=30,
                        help="Frames to wait before counting another violation from the same vehicle")
    parser.add_argument("--violation_classes", type=str, default="all",
                        help="Comma-separated list of class indices to monitor for violations, or 'all'")
    parser.add_argument("--min_frames_in_green", type=int, default=3,
                        help="Minimum frames vehicle must be tracked in green zone before violation is counted")
    parser.add_argument("--trajectory_interpolation", type=int, default=5,
                        help="Number of interpolation steps for trajectory (for fast moving vehicles)")
    
    # Violation detection method
    parser.add_argument("--detection_method", type=str, default="combined", 
                        choices=["stop_line", "zone", "combined"],
                        help="Method to use for violation detection")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug output")
    
    # Convert to TensorRT
    parser.add_argument("--use_tensorrt", action="store_true", help="Chuyển đổi mô hình sang TensorRT để tăng tốc inference")
    parser.add_argument("--half_precision", action="store_true", help="Sử dụng FP16 cho TensorRT (tăng tốc hơn nữa)")
    parser.add_argument("--tensorrt_workspace", type=int, default=8, help="Giới hạn workspace (GB) cho TensorRT")
    parser.add_argument("--tensorrt_dynamic", action="store_true", help="Sử dụng kích thước batch động cho TensorRT")
    
    return parser.parse_args()


def deblur(nafnet, img, device, memory_efficient=False, max_size=1280):
    """
    Apply deblurring using NAFNet with memory optimization
    """
    # Resize large images to conserve memory
    orig_h, orig_w = img.shape[:2]
    if max(orig_h, orig_w) > max_size:
        scale = max_size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        img = cv2.resize(img, (new_w, new_h))
        resized = True
    else:
        resized = False
    
    # Convert to RGB and normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # Convert to tensor
    img_tensor = _img2tensor(img, bgr2rgb=False, float32=True)
    
    # Move to CPU if memory efficient mode is on
    if memory_efficient:
        img_tensor = img_tensor.cpu()
    
    # Process image
    try:
        nafnet.feed_data(data={"lq": img_tensor.unsqueeze(dim=0)})
        
        if nafnet.opt["val"].get("grids", False):
            nafnet.grids()
            
        # Clear cache before the heaviest operation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Run the deblurring
        nafnet.test()
        
        if nafnet.opt["val"].get("grids", False):
            nafnet.grids_inverse()
            
        # Get and convert result
        visuals = nafnet.get_current_visuals()
        sr_img = tensor2img([visuals["result"]])
        
        # Resize back to original size if needed
        if resized:
            sr_img = cv2.resize(sr_img, (orig_w, orig_h))
            
        return sr_img
    
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"Warning: Not enough memory for deblurring, returning original image")
            return cv2.cvtColor(img * 255, cv2.COLOR_RGB2BGR).astype(np.uint8)
        else:
            raise


def infer_classify(model, source, device) -> int:
    """Classify an image with memory optimization"""
    try:
        classify_results = model.predict(
            source=source,
            verbose=False,
            device=device
        )[0]
        cls_ = classify_results.probs.top1
        return cls_
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("Warning: CUDA out of memory for classification, falling back to CPU")
            classify_results = model.predict(
                source=source,
                verbose=False,
                device="cpu"
            )[0]
            cls_ = classify_results.probs.top1
            return cls_
        else:
            raise


def infer_detect_track(model, source, conf=0.01, iou=0.7, device="cuda", tracker_type="bytetrack", persist=True):
    """Detect and track objects using ultralytics built-in tracker"""
    try:
        results = model.track(
            source=source,
            conf=conf,
            iou=iou,
            verbose=False,
            device=device,
            tracker="bytetrack.yaml", 
            persist=persist       
        )[0]
        return results
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("Warning: CUDA out of memory for detection/tracking, falling back to CPU")
            results = model.track(
                source=source,
                conf=conf,
                iou=iou,
                verbose=False,
                device="cpu",
                tracker=tracker_type,
                persist=persist
            )[0]
            return results
        else:
            raise


def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using ray casting algorithm
    
    Parameters:
    - point: (x, y) tuple, the point to check
    - polygon: [(x1, y1), (x2, y2), ...] list of points forming the polygon
    
    Returns:
    - Boolean, True if the point is inside the polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def line_intersection(line1, line2):
    """
    Determine if two line segments intersect
    line1 and line2 are in format (x1, y1, x2, y2)
    Returns True if the lines intersect, False otherwise
    """
    # Convert line segments to parametric form
    def line_to_params(line):
        x1, y1, x2, y2 = line
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        return A, B, C
    
    A1, B1, C1 = line_to_params(line1)
    A2, B2, C2 = line_to_params(line2)
    
    # Check if lines are parallel
    det = A1 * B2 - A2 * B1
    if det == 0:
        return False, None
    
    # Find intersection point
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    
    # Check if intersection point is within both line segments
    def is_between(a, b, c):
        # Check if c is between a and b with some tolerance
        # to account for floating point errors
        margin = 1e-9
        return (min(a, b) - margin <= c <= max(a, b) + margin)
    
    if (is_between(line1[0], line1[2], x) and 
        is_between(line1[1], line1[3], y) and 
        is_between(line2[0], line2[2], x) and 
        is_between(line2[1], line2[3], y)):
        return True, (x, y)
    
    return False, None


def is_red_light_phase(frame_count, red_light_ranges=None, always_red=False):
    """
    Determine if the current frame is in a red light phase
    
    Parameters:
    - frame_count: Current frame number
    - red_light_ranges: List of (start, end) tuples defining red light phases
    - always_red: If True, always return True (simulating constant red light)
    
    Returns:
    - Boolean, True if the current frame is in a red light phase
    """
    if always_red:
        return True
        
    if red_light_ranges is None:
        return False
        
    for start, end in red_light_ranges:
        if start <= frame_count <= end:
            return True
            
    return False


def check_stop_line_violation(prev_pos, curr_pos, stop_line=None, stop_y=None, tolerance=10):
    """
    Check if a vehicle crossed the stop line between previous and current position
    
    Parameters:
    - prev_pos: (x, y) tuple, previous position
    - curr_pos: (x, y) tuple, current position
    - stop_line: (x1, y1, x2, y2) tuple, stop line coordinates (priority)
    - stop_y: Y-coordinate of horizontal stop line (used if stop_line is None)
    - tolerance: Tolerance in pixels for the stop line
    
    Returns:
    - crossed: Boolean, True if crossed the line
    - cross_point: (x, y) approximate crossing point or None
    """
    if stop_line is not None:
        # Define the line segment from previous to current position
        movement_line = (prev_pos[0], prev_pos[1], curr_pos[0], curr_pos[1])
        
        # Check if the segments intersect
        crossed, cross_point = line_intersection(movement_line, stop_line)
        return crossed, cross_point
    
    elif stop_y is not None:
        # Simple horizontal stop line
        prev_x, prev_y = prev_pos
        curr_x, curr_y = curr_pos
        
        # Check if trajectory crosses the stop line
        if ((prev_y < stop_y - tolerance and curr_y > stop_y + tolerance) or
            (prev_y > stop_y + tolerance and curr_y < stop_y - tolerance)):
            
            # Calculate approximate crossing point (linear interpolation)
            if prev_y != curr_y:  # Avoid division by zero
                t = (stop_y - prev_y) / (curr_y - prev_y)
                cross_x = prev_x + t * (curr_x - prev_x)
                return True, (cross_x, stop_y)
            else:
                return True, (prev_x, stop_y)
        
        return False, None
    
    return False, None


def check_zone_transition(track, green_zone, red_zone, min_frames_in_green=3, debug=False):
    """
    Check if a vehicle moved from green zone to red zone
    
    Parameters:
    - track: Dictionary with vehicle tracking data
    - green_zone: List of (x,y) points defining the legal approach zone
    - red_zone: List of (x,y) points defining the violation zone
    - min_frames_in_green: Minimum frames vehicle must be in green zone
    
    Returns:
    - violated: Boolean, True if violation detected
    - info: Dictionary with violation details
    """
    # Ensure we have enough history
    if "history" not in track or len(track["history"]) < 2:
        return False, {}
    
    # Check if zones are defined
    if red_zone is None or green_zone is None:
        return False, {}
    
    # Get the current position (most recent point in history)
    current_pos = track["history"][-1]
    in_red_zone = point_in_polygon(current_pos, red_zone)
    
    # Initialize zone tracking if not present
    if "zone_history" not in track:
        track["zone_history"] = []
        track["frames_in_green"] = 0
        track["zone_violated"] = False
    
    # Get current zone
    current_zone = None
    if point_in_polygon(current_pos, green_zone):
        current_zone = "green"
        track["frames_in_green"] += 1
    elif point_in_polygon(current_pos, red_zone):
        current_zone = "red"
    else:
        current_zone = "outside"
    
    # Update zone history
    track["zone_history"].append(current_zone)
    
    # Keep only the last 50 zone entries
    if len(track["zone_history"]) > 50:
        track["zone_history"] = track["zone_history"][-50:]
    
    # Check for green -> red transition
    if in_red_zone and not track["zone_violated"]:
        # Check if vehicle was previously in green zone long enough
        if track["frames_in_green"] >= min_frames_in_green:
            if debug:
                print(f"DEBUG: Track {track.get('track_id', 0)} was in green zone for {track['frames_in_green']} frames, now in red zone")
            track["zone_violated"] = True
            violation_info = {
                "track_id": track.get("track_id", 0),
                "violation_type": "zone_transition",
                "frames_in_green": track["frames_in_green"]
            }
            return True, violation_info
    
    return False, {}


def interpolate_trajectory(p1, p2, steps=5):
    """
    Create interpolated points between two trajectory points
    
    Parameters:
    - p1: (x1, y1) tuple, start point
    - p2: (x2, y2) tuple, end point
    - steps: Number of interpolation steps (including endpoints)
    
    Returns:
    - List of (x, y) points including original endpoints
    """
    points = []
    for i in range(steps):
        t = i / (steps - 1)
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        points.append((x, y))
    return points


def check_violation_with_interpolation(track, frame_count, green_zone, red_zone, 
                                      stop_line=None, stop_y=None, tolerance=10,
                                      detection_method="combined", 
                                      min_frames_in_green=3, 
                                      interpolation_steps=5,
                                      debug=False):
    """
    Advanced violation detection with interpolation for fast moving vehicles
    
    Parameters:
    - track: Dictionary with vehicle tracking data
    - frame_count: Current frame number
    - green_zone: List of (x,y) points defining the legal approach zone
    - red_zone: List of (x,y) points defining the violation zone
    - stop_line/stop_y: Stop line parameters
    - detection_method: 'stop_line', 'zone', or 'combined'
    - interpolation_steps: Number of points to interpolate for trajectory
    
    Returns:
    - violated: Boolean, True if violation detected
    - info: Dictionary with violation details
    """
    # Ensure we have enough history
    if "history" not in track or len(track["history"]) < 2:
        return False, {}
    
    # Get the previous and current positions
    prev_pos = track["history"][-2]
    curr_pos = track["history"][-1]
    
    # Create an interpolated trajectory for more accurate checking
    interpolated_points = interpolate_trajectory(prev_pos, curr_pos, interpolation_steps)
    
    # Initialize violation flags
    stop_line_violated = False
    zone_violated = False
    crossing_point = None
    
    # Method 1: Check stop line violation
    if detection_method in ["stop_line", "combined"]:
        # Check pairs of consecutive interpolated points
        for i in range(1, len(interpolated_points)):
            p1 = interpolated_points[i-1]
            p2 = interpolated_points[i]
            
            crossed, cross_point = check_stop_line_violation(
                p1, p2, stop_line, stop_y, tolerance
            )
            
            if crossed:
                stop_line_violated = True
                crossing_point = cross_point
                break
    
    # Method 2: Check zone transition
    if detection_method in ["zone", "combined"]:
        zone_result, zone_info = check_zone_transition(
            track, green_zone, red_zone, min_frames_in_green, debug
        )
        
        if zone_result:
            zone_violated = True
    
    # Determine overall violation based on detection method
    violation_occurred = False
    violation_info = {}
    
    if detection_method == "stop_line":
        violation_occurred = stop_line_violated
        if violation_occurred:
            violation_info = {
                "track_id": track.get("track_id", 0),
                "violation_type": "stop_line",
                "crossing_point": crossing_point
            }
    elif detection_method == "zone":
        violation_occurred = zone_violated
        if violation_occurred:
            violation_info = zone_info
    else:  # combined
        violation_occurred = stop_line_violated or zone_violated
        if violation_occurred:
            violation_info = {
                "track_id": track.get("track_id", 0),
                "violation_type": "stop_line" if stop_line_violated else "zone_transition",
                "crossing_point": crossing_point
            }
    
    return violation_occurred, violation_info

def visualize_tracked_image(image, detections, trajectories=None, trajectory_length=30,
                           stop_line=None, stop_y=None, green_zone=None, red_zone=None,
                           violation_count=0, is_red_phase=False, violations=None, 
                           detection_method="combined", frame_count=0, debug=False):
    """
    Visualize detections with tracking information, zones and violations - simplified version
    """
    
    h, w = image.shape[:2]
    
    # Define colors for different vehicle classes
    colors = {
        0: (0, 255, 0),    # Green 
        1: (255, 0, 0),    # Blue
        2: (0, 0, 255),    # Red 
        3: (255, 255, 0),  # Cyan 
    }
    
    # Create a copy of the image for overlay
    overlay = image.copy()
    
    # Draw red and green zones if provided
    if red_zone:
        # Convert to numpy array for drawing
        pts = np.array(red_zone, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Draw filled polygon with transparency
        cv2.fillPoly(overlay, [pts], (0, 0, 255))  # Red with alpha
    
    if green_zone:
        # Convert to numpy array for drawing
        pts = np.array(green_zone, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Draw filled polygon with transparency
        cv2.fillPoly(overlay, [pts], (0, 255, 0))  # Green with alpha
    
    # Add overlay with transparency
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
    
    # Draw stop line if provided
    if stop_line is not None:
        x1, y1, x2, y2 = stop_line
        # Change color based on light phase: red or green
        line_color = (0, 0, 255) if is_red_phase else (0, 255, 0)  # Red or Green
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), line_color, 3)
    
    elif stop_y is not None:
        # Draw horizontal stop line
        line_color = (0, 0, 255) if is_red_phase else (0, 255, 0)  # Red or Green
        cv2.line(image, (0, stop_y), (w, stop_y), line_color, 3)
    
    # Add traffic light status indicator in corner
    status_color = (0, 0, 255) if is_red_phase else (0, 255, 0)  # Red or Green
    cv2.rectangle(image, (w-150, 10), (w-10, 50), status_color, -1)  # Filled rectangle
    status_text = "RED LIGHT" if is_red_phase else "GREEN LIGHT"
    cv2.putText(image, status_text, (w-140, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add violation counter with prominent display
    cv2.rectangle(image, (5, 25), (220, 60), (0, 0, 0), -1)  # Black background
    cv2.putText(image, f"VIOLATIONS: {violation_count}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Draw bounding boxes and labels (simplified)
    for i, (box, label, score, track_id, speed) in enumerate(detections):
        x1, y1, x2, y2 = box.astype(int)
        
        # Check if this vehicle has violated
        has_violated = False
        if violations is not None and track_id in violations:
            has_violated = True
        
        # Use different color for violators
        if has_violated:
            color = (0, 0, 255)  # Red for violators
            # Draw a thicker box for violators
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            # Add "VIOLATOR" label
            cv2.putText(image, "VIOLATION", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            color = colors.get(int(label), (0, 255, 0))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Simplified label with just track ID
            cv2.putText(image, f"ID:{track_id}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw trajectories if available (simplified)
    if trajectories:
        for track_id, track in trajectories.items():
            if "history" not in track or len(track["history"]) < 2:
                continue
                
            # Check if this track has a violation
            has_violated = violations is not None and track_id in violations
            
            # Get color for this track - red for violators
            if has_violated:
                color = (0, 0, 255)  # Red for violators
            else:
                label = track.get("label", 0)
                color = colors.get(int(label), (0, 255, 0))
            
            # Draw the recent trajectory (limited by trajectory_length)
            history = track["history"][-trajectory_length:] if len(track["history"]) > trajectory_length else track["history"]
            for i in range(1, len(history)):
                # Convert center points to integers
                pt1 = (int(history[i-1][0]), int(history[i-1][1]))
                pt2 = (int(history[i][0]), int(history[i][1]))
                
                # Ensure points are within image bounds
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    # Make violator trajectories thicker
                    thickness = 3 if has_violated else 2
                    cv2.line(image, pt1, pt2, color, thickness)
            
            # Mark violation point for violators
            if has_violated and "crossing_point" in violations[track_id] and violations[track_id]["crossing_point"] is not None:
                cross_x, cross_y = violations[track_id]["crossing_point"]
                cv2.drawMarker(image, (int(cross_x), int(cross_y)), (0, 0, 255),
                              markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
    
    return image

def convert_to_tensorrt(model, model_path, half=False, workspace=8, dynamic=False, device=0, task=None):
    """
    Chuyển đổi mô hình YOLO sang TensorRT
    
    Args:
        model: YOLO model instance
        model_path: Đường dẫn đến mô hình gốc
        half: Sử dụng half precision (FP16) hay không
        workspace: Giới hạn workspace GB cho TensorRT
        dynamic: Kích thước batch động hay không
        device: GPU device index
        task: Loại nhiệm vụ của mô hình ('detect', 'classify', etc.)
        
    Returns:
        Đường dẫn đến mô hình TensorRT đã chuyển đổi
    """
    # Define task
    if task is None:
        if "vehicle" in model_path.lower():
            task = "detect"
        elif "day_night" in model_path.lower() or "classify" in model_path.lower():
            task = "classify"
        else:
            task = "detect" 
    
    print(f"Chuyển đổi {model_path} sang định dạng TensorRT{'(FP16)' if half else ''} cho task '{task}'")

    basename = os.path.splitext(model_path)[0]
    engine_path = f"{basename}_{task}.engine"
    
    if os.path.exists(engine_path):
        print(f"Đã tìm thấy TensorRT engine tại {engine_path}, sử dụng file có sẵn")
        return engine_path
    
    # Export
    try:
        model.export(format='engine', 
                     half=half, 
                     workspace=workspace, 
                     device=device,
                     dynamic=dynamic)
        
        default_export_path = model_path.replace('.pt', '.engine')
        
        if os.path.exists(default_export_path) and default_export_path != engine_path:
            os.rename(default_export_path, engine_path)
            
        if os.path.exists(engine_path):
            print(f"Chuyển đổi thành công, engine lưu tại: {engine_path}")
            return engine_path
        else:
            print(f"Không tìm thấy engine sau khi chuyển đổi")
            return None
    except Exception as e:
        print(f"Lỗi khi chuyển đổi sang TensorRT: {str(e)}")
        print("Quay trở lại sử dụng mô hình PyTorch")
        return None

def check_tensorrt_compatibility():
    """Kiểm tra khả năng tương thích với TensorRT"""
    if not torch.cuda.is_available():
        print("TensorRT yêu cầu GPU NVIDIA. Không tìm thấy GPU CUDA.")
        return False
    
    try:
        import tensorrt
        print(f"Phiên bản TensorRT: {tensorrt.__version__}")
        return True
    except ImportError:
        print("TensorRT chưa được cài đặt. Hãy cài đặt tensorrt package.")
        return False

def calculate_speed(track, fps, pixels_per_meter=10):
    """
    Estimate speed in km/h based on trajectory and framerate
    """
    if "history" not in track or len(track["history"]) < 2:
        return 0
    
    # Calculate distance in pixels between last two positions
    p1 = track["history"][-2]
    p2 = track["history"][-1]
    distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    # Convert to meters
    distance_meters = distance_pixels / pixels_per_meter
    
    # Calculate time between frames in hours (fps is frames per second)
    time_hours = 1 / (fps * 3600)
    
    # Calculate speed in km/h
    speed_kmh = (distance_meters / 1000) / time_hours
    
    return min(speed_kmh, 150)  


def process_video():
    ## Parse arguments ----------------------------------------------
    args = parse_arguments()
    
    if args.use_tensorrt:
        tensorrt_available = check_tensorrt_compatibility()
        if not tensorrt_available:
            print("Chuyển về chế độ PyTorch vì TensorRT không khả dụng")
            args.use_tensorrt = False
    
    # Check if process_resolution is specified
    target_width, target_height = None, None
    if args.process_resolution:
        try:
            target_width, target_height = map(int, args.process_resolution.split('x'))
        except:
            print(f"Invalid resolution format: {args.process_resolution}. Using original resolution.")
    
    output_video_path = args.output_video
    output_data_path = args.output_data
    
    # Parse stop line parameters
    stop_line = None
    stop_y = args.stop_y
    
    if args.stop_line:
        try:
            stop_line = tuple(map(float, args.stop_line.split(',')))
            if len(stop_line) != 4:
                print("Invalid stop line format. Must be x1,y1,x2,y2")
                stop_line = None
            else:
                print(f"Stop line set at: {stop_line}")
        except:
            print(f"Error parsing stop line: {args.stop_line}")
            stop_line = None
    
    # Parse red and green zones
    green_zone = None
    red_zone = None
    
    if args.green_zone:
        try:
            coords = list(map(float, args.green_zone.split(',')))
            if len(coords) < 6 or len(coords) % 2 != 0:
                print("Invalid green zone format. Must have at least 3 points (6 coordinates)")
            else:
                # Convert to list of (x, y) points
                green_zone = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                print(f"Green zone set with {len(green_zone)} points")
        except:
            print(f"Error parsing green zone: {args.green_zone}")
    
    if args.red_zone:
        try:
            coords = list(map(float, args.red_zone.split(',')))
            if len(coords) < 6 or len(coords) % 2 != 0:
                print("Invalid red zone format. Must have at least 3 points (6 coordinates)")
            else:
                # Convert to list of (x, y) points
                red_zone = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                print(f"Red zone set with {len(red_zone)} points")
        except:
            print(f"Error parsing red zone: {args.red_zone}")
    
    # Parse red light phases
    red_light_ranges = []
    if args.red_light_frames:
        try:
            frames = list(map(int, args.red_light_frames.split(',')))
            if len(frames) % 2 != 0:
                print("Invalid red light frames format. Must be start1,end1,start2,end2,...")
            else:
                red_light_ranges = [(frames[i], frames[i+1]) for i in range(0, len(frames), 2)]
                print(f"Red light phases: {red_light_ranges}")
        except:
            print(f"Error parsing red light frames: {args.red_light_frames}")
    
    # Parse violation classes
    violation_classes = []
    if args.violation_classes != "all":
        try:
            violation_classes = list(map(int, args.violation_classes.split(',')))
            print(f"Monitoring classes for violations: {violation_classes}")
        except:
            print(f"Error parsing violation classes: {args.violation_classes}. Monitoring all classes.")
            violation_classes = []
    
    # Create preview directory if needed
    preview_dir = None
    if args.save_preview_frames > 0:
        preview_dir = "preview_frames"
        os.makedirs(preview_dir, exist_ok=True)
        print(f"Will save preview frames to {preview_dir}/")
    
    ## Load models -------------------------------------------------
    print("Loading models...")
    
    # Set device based on availability and memory_efficient flag
    if args.memory_efficient:
        # Force CPU for all models in memory efficient mode
        device = torch.device("cpu")
        enhancement_device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enhancement_device = device
    
    # Load vehicle detection model
    vehicle_model = YOLO(args.vehicle_model_path, task="detect")
    
    # Load day/night classification model
    daynight_model = YOLO(args.daynight_model_path, task="classify")
    
    if args.use_tensorrt and torch.cuda.is_available():
        print("Đang chuyển đổi mô hình sang TensorRT...")
        
    # TensorRT
        vehicle_engine_path = convert_to_tensorrt(
            vehicle_model,
            args.vehicle_model_path, 
            half=args.half_precision,
            workspace=args.tensorrt_workspace,
            dynamic=args.tensorrt_dynamic,
            device=0 if device.type == "cuda" else "cpu",
            task="detect"
        )
        
        if vehicle_engine_path:
            vehicle_model = YOLO(vehicle_engine_path, task="detect")
            print("Đã tải mô hình xe cộ từ TensorRT engine")
        
        # TensorRT
        daynight_engine_path = convert_to_tensorrt(
            daynight_model,
            args.daynight_model_path,
            half=args.half_precision,
            workspace=args.tensorrt_workspace,
            dynamic=args.tensorrt_dynamic,
            device=0 if device.type == "cuda" else "cpu",
            task="classify"
        )
        
        if daynight_engine_path:
            daynight_model = YOLO(daynight_engine_path, task="classify")
            print("Đã tải mô hình phân loại ngày/đêm từ TensorRT engine")
    
    # Load enhancement models if not disabled
    if not args.disable_enhancement:
        light_enhancer = LoLi_IEA(args.loli_iea_model_dir, enhancement_device)
        
        # Configure NAFNet based on device
        opt_path = "weights/NAFNNet/NAFNet-width64.yml"
        opt = parse(opt_path, is_train=False)
        opt["dist"] = False
        
        # Set device in the options
        if enhancement_device.type == "cpu":
            opt["num_gpus"] = 0 
        else:
            opt["num_gpus"] = 1
            
        # Create model with the appropriate device settings
        NAFNet = create_model(opt)
    else:
        light_enhancer = None
        NAFNet = None
    
    # Initialize trajectory storage
    trajectories = {} 
    vehicle_data = defaultdict(list)  
    
    # Initialize violation tracking
    violation_count = 0
    violations = {} 
    
    ## Warmup models -----------------------------------------------
    print("Warming up models...")
    dummy_detect_frame = np.zeros((640, 640, 3), dtype=np.uint8)  
    dummy_classify_frame = np.zeros((224, 224, 3), dtype=np.uint8) 

    try:
        vehicle_model.predict(dummy_detect_frame, device=device, verbose=False)
        daynight_model.predict(dummy_classify_frame, device=device, verbose=False)
    except Exception as e:
        print(f"Cảnh báo: Lỗi khi khởi động mô hình: {e}")
        print("Tiếp tục chạy mặc dù có lỗi warmup...")
    
    ## Set up video capture ----------------------------------------
    if args.img_dir is None:
        # Process video
        try:
            # Check if input is a camera index
            if args.input.isdigit():
                cap = cv2.VideoCapture(int(args.input))
            else:
                cap = cv2.VideoCapture(args.input)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video source: {args.input}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Override with target resolution if specified
            if target_width and target_height:
                width, height = target_width, target_height
            
            # Check if we need to resize based on max_dimension
            if max(width, height) > args.max_dimension:
                scale_factor = args.max_dimension / max(width, height)
                width = int(width * scale_factor)
                height = int(height * scale_factor)
                print(f"Resizing frames to {width}x{height} to stay within max dimension of {args.max_dimension}")
            
            # Set default zones and stop line if not provided
            if not args.stop_line and not args.stop_y and not args.red_zone and not args.green_zone:
                print("No zones or stop line provided. Using default setup.")
                stop_y = height // 2
                
                # Create default zones based on stop line
                green_zone = [(0, stop_y+10), (width, stop_y+10), (width, height), (0, height)]
                red_zone = [(0, 0), (width, 0), (width, stop_y-10), (0, stop_y-10)]
                
                print(f"Default stop line at y={stop_y}")
                print(f"Default green zone (below stop line): {green_zone}")
                print(f"Default red zone (above stop line): {red_zone}")
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # Initialize data file
            with open(output_data_path, 'w') as f:
                f.write("frame,track_id,class,x1,y1,x2,y2,confidence,speed_kmh,violation\n")
            
            # Process video frames
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if args.input.isdigit() is False else float('inf')
            
            # Set up visualization (if not in headless mode)
            show_visualization = args.vis and not args.headless
            display_available = True
            
            # Try to determine if display is available (for X11 issues)
            if show_visualization:
                try:
                    # Test if we can create a window
                    cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
                    cv2.destroyWindow("Test Window")
                except:
                    print("Warning: Could not create display window. Running in headless mode.")
                    display_available = False
                    show_visualization = False
            
            # Progress bar
            pbar = tqdm(total=total_frames, desc="Processing video")
            
            # Main processing loop
            while True:
                # Clear CUDA cache to prevent memory issues
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed
                if args.skip_frames > 0 and frame_count % (args.skip_frames + 1) != 0:
                    frame_count += 1
                    pbar.update(1)
                    continue
                
                # Resize if needed
                if target_width and target_height:
                    frame = cv2.resize(frame, (target_width, target_height))
                elif max(frame.shape[0], frame.shape[1]) > args.max_dimension:
                    scale_factor = args.max_dimension / max(frame.shape[0], frame.shape[1])
                    new_width = int(frame.shape[1] * scale_factor)
                    new_height = int(frame.shape[0] * scale_factor)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Check if current frame is in a red light phase
                is_red_phase = is_red_light_phase(
                    frame_count, 
                    red_light_ranges, 
                    always_red=args.red_light
                )
                
                # Process frame
                try:
                    processed_frame, frame_detections, new_violations = process_frame(
                        frame, frame_count, vehicle_model, daynight_model, 
                        NAFNet, light_enhancer, device, enhancement_device,
                        args.conf, args.iou, trajectories, vehicle_data, fps,
                        args.memory_efficient, args.disable_enhancement, args.max_dimension,
                        args.trajectory_length, args.tracker,
                        stop_line, stop_y, args.tolerance, green_zone, red_zone,
                        is_red_phase, violations, violation_count, 
                        args.detection_method, violation_classes, args.violation_cooldown,
                        args.min_frames_in_green, args.trajectory_interpolation,
                        args.debug
                    )
                    
                    # Update violation count
                    violation_count += new_violations
                    
                    # Write results to output file
                    with open(output_data_path, 'a') as f:
                        for det in frame_detections:
                            track_id, cls, x1, y1, x2, y2, conf, speed, is_violation = det
                            f.write(f"{frame_count},{track_id},{cls},{x1},{y1},{x2},{y2},{conf},{speed},{1 if is_violation else 0}\n")
                    
                    # Write frame to output video
                    out.write(processed_frame)
                    
                    # Save preview frames if requested
                    if preview_dir and args.save_preview_frames > 0 and frame_count % args.save_preview_frames == 0:
                        preview_path = os.path.join(preview_dir, f"frame_{frame_count:06d}.jpg")
                        cv2.imwrite(preview_path, processed_frame)
                    
                    # Display if visualization is enabled
                    if show_visualization and display_available:
                        try:
                            cv2.imshow('Processed Frame', processed_frame)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                break
                            elif key == ord('p'):
                                # Pause/play on 'p' key
                                cv2.waitKey(0)
                        except Exception as e:
                            print(f"Display error: {e}. Continuing in headless mode.")
                            display_available = False
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Write original frame on error
                    try:
                        out.write(frame)
                    except:
                        print("Error writing original frame")
                
                frame_count += 1
                pbar.update(1)
            
            # Clean up
            cap.release()
            out.release()
            
            # Only try to close windows if display is available
            if display_available and show_visualization:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass
                
            pbar.close()
            
            print(f"Total violations detected: {violation_count}")
            
        except Exception as e:
            print(f"Error processing video: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Process images from directory
        process_image_directory(args, device, enhancement_device, 
                               vehicle_model, daynight_model, NAFNet, light_enhancer,
                               stop_line, stop_y, args.tolerance, green_zone, red_zone)
    
    print(f"Processing complete. Results saved to {output_video_path} and {output_data_path}")
    print(f"Total violations detected: {violation_count}")


def process_frame(frame, frame_count, vehicle_model, daynight_model, NAFNet, light_enhancer,
                 device, enhancement_device, conf_threshold, iou_threshold, 
                 trajectories, vehicle_data, fps, memory_efficient=False,
                 disable_enhancement=False, max_dimension=1280, trajectory_length=30,
                 tracker_type="bytetrack", stop_line=None, stop_y=None, tolerance=10,
                 green_zone=None, red_zone=None, is_red_phase=False, violations=None, 
                 violation_count=0, detection_method="combined", violation_classes=None,
                 violation_cooldown=30, min_frames_in_green=3, interpolation_steps=5,
                 debug=False):
    """Process a single frame, returning the processed frame and detection data"""
    # Clear CUDA cache to prevent memory issues
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Copy original frame for visualization
    original_frame = frame.copy()
    processed_frame = None
    
    # Initialize violations dict if not provided
    if violations is None:
        violations = {}
    
    # Track new violations in this frame
    new_violations = 0
    
    try:
        # Enhance the image if enhancement is enabled
        if not disable_enhancement:
            # Classify as day or night
            day_night_cls = infer_classify(daynight_model, frame, 
                                          "cpu" if memory_efficient else device)
            
            # Apply appropriate enhancement
            if day_night_cls == 0: 
                processed_frame = deblur(NAFNet, frame, device, memory_efficient, max_dimension)
            else: 
                try:
                    processed_frame = light_enhancer.enhance_image(frame)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"Warning: CUDA out of memory for light enhancement, using original frame")
                        processed_frame = frame.copy()
                    else:
                        raise
        else:
            # Skip enhancement
            processed_frame = frame.copy()
        
        # Detect vehicles
        if processed_frame is not None:
            detect_frame = processed_frame
        else:
            detect_frame = frame
        
        # Run vehicle detection and tracking
        detect_device = "cpu" if memory_efficient else device
        results = infer_detect_track(
            model=vehicle_model,
            source=detect_frame,
            conf=conf_threshold,
            iou=iou_threshold,
            device=detect_device,
            tracker_type=tracker_type,
            persist=True  # Keep tracking between frames
        )
        
        # Process tracking results
        if hasattr(results, 'boxes') and len(results.boxes) > 0:
            # Check if tracking is enabled and track IDs are available
            if hasattr(results.boxes, 'id') and results.boxes.id is not None:
                # Get boxes, track IDs, labels, and confidence scores
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.int().cpu().numpy()
                labels = results.boxes.cls.int().cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                
                # Create detection records for this frame
                frame_detections = []
                viz_detections = []
                
                for i, (box, track_id, label, score) in enumerate(zip(boxes, track_ids, labels, scores)):
                    # Calculate center point for trajectory
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    
                    # Check if this class should be monitored for violations
                    is_monitored = len(violation_classes) == 0 or int(label) in violation_classes
                    
                    is_violation = False
                    
                    # Create or update trajectory data
                    if track_id not in trajectories:
                        trajectories[track_id] = {
                            "history": [(center_x, center_y)],
                            "label": label,
                            "first_seen": frame_count,
                            "last_seen": frame_count,
                            "track_id": int(track_id)
                        }
                    else:
                        # Update trajectory
                        trajectories[track_id]["history"].append((center_x, center_y))
                        trajectories[track_id]["last_seen"] = frame_count
                        
                        # Limit history length
                        if len(trajectories[track_id]["history"]) > 50:
                            trajectories[track_id]["history"] = trajectories[track_id]["history"][-50:]
                        
                        # Check for violation (only during red light phase)
                        if is_red_phase and is_monitored and track_id not in violations:
                            # Advanced violation check with interpolation
                            violated, violation_info = check_violation_with_interpolation(
                                trajectories[track_id], frame_count, 
                                green_zone, red_zone, 
                                stop_line, stop_y, tolerance,
                                detection_method, min_frames_in_green, 
                                interpolation_steps, debug
                            )
                            
                            if violated:
                                # Get violation type for display
                                violation_type = violation_info.get("violation_type", "unknown")
                                
                                # Debug output
                                if debug:
                                    print(f"🚨 VIOLATION DETECTED! Vehicle {track_id}, frame {frame_count}, type: {violation_type}")
                                else:
                                    print(f"🚨 VIOLATION DETECTED! Vehicle {track_id}, frame {frame_count}")
                                
                                is_violation = True
                                new_violations += 1
                                
                                # Store violation info
                                violations[track_id] = {
                                    "frame": frame_count,
                                    "type": violation_type,
                                    "crossing_point": violation_info.get("crossing_point", None)
                                }
                    
                    # Calculate speed
                    speed = calculate_speed(trajectories[track_id], fps)
                    
                    # Store detection data
                    vehicle_data[track_id].append({
                        "frame": frame_count,
                        "box": box,
                        "label": label,
                        "score": score,
                        "speed": speed,
                        "violation": is_violation
                    })
                    
                    # Add to frame detections for output
                    frame_detections.append((
                        int(track_id), int(label), 
                        int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                        score, speed, is_violation or track_id in violations
                    ))
                    
                    # Add to visualization list
                    viz_detections.append((
                        box, label, score, int(track_id), speed
                    ))
                
                # Visualize detections with trajectories, zones and stop line
                result_frame = visualize_tracked_image(
                    original_frame, viz_detections, trajectories, trajectory_length,
                    stop_line, stop_y, green_zone, red_zone,
                    violation_count + new_violations, is_red_phase, violations,
                    detection_method, frame_count, debug
                )
            else:
                # If tracking is not enabled or failed, just show detections
                print("Warning: Tracking not available for this frame. Using detection only.")
                boxes = results.boxes.xyxy.cpu().numpy()
                labels = results.boxes.cls.int().cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                
                viz_detections = []
                frame_detections = []
                
                for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                    # Use dummy track IDs
                    dummy_id = i + 1
                    
                    # Add to visualization list
                    viz_detections.append((
                        box, label, score, dummy_id, 0.0
                    ))
                    
                    # Add to frame detections
                    frame_detections.append((
                        dummy_id, int(label),
                        int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                        score, 0.0, False  # No violation info without tracking
                    ))
                
                result_frame = visualize_tracked_image(
                    original_frame, viz_detections, None, trajectory_length,
                    stop_line, stop_y, green_zone, red_zone,
                    violation_count, is_red_phase, violations,
                    detection_method, frame_count, debug
                )
        else:
            # No detections
            result_frame = original_frame
            
            # Still draw the zones and stop line
            h, w = result_frame.shape[:2]
            
            # Create overlay image for zones
            overlay = result_frame.copy()
            
            # Draw red and green zones if provided
            if red_zone:
                pts = np.array(red_zone, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], (0, 0, 255))  # Red
            
            if green_zone:
                pts = np.array(green_zone, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], (0, 255, 0))  # Green
            
            # Add overlay with transparency
            cv2.addWeighted(overlay, 0.3, result_frame, 0.7, 0, result_frame)
            
            # Draw stop line if provided
            if stop_line is not None:
                x1, y1, x2, y2 = stop_line
                # Change color based on light phase: red or green
                line_color = (0, 0, 255) if is_red_phase else (0, 255, 0)  # Red or Green
                cv2.line(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), line_color, 3)
            
            elif stop_y is not None:
                # Draw horizontal stop line
                line_color = (0, 0, 255) if is_red_phase else (0, 255, 0)  # Red or Green
                cv2.line(result_frame, (0, stop_y), (w, stop_y), line_color, 3)
            
            # Add traffic light status indicator in corner
            status_color = (0, 0, 255) if is_red_phase else (0, 255, 0)  # Red or Green
            cv2.rectangle(result_frame, (w-150, 10), (w-10, 50), status_color, -1)  # Filled rectangle
            status_text = "RED LIGHT" if is_red_phase else "GREEN LIGHT"
            cv2.putText(result_frame, status_text, (w-140, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add violation counter
            cv2.rectangle(result_frame, (5, 95), (270, 125), (0, 0, 0), -1)  # Black background
            cv2.putText(result_frame, f"VIOLATIONS: {violation_count}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            frame_detections = []
    
    except Exception as e:
        print(f"Error processing frame {frame_count}: {e}")
        import traceback
        traceback.print_exc()
        result_frame = original_frame
        frame_detections = []
    
    # Add counter for tracked vehicles
    active_vehicles = len([tid for tid, track in trajectories.items() 
                          if track["last_seen"] >= frame_count - 30])
    cv2.putText(result_frame, f"Vehicles: {active_vehicles}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return result_frame, frame_detections, new_violations


def process_image_directory(args, device, enhancement_device, 
                           vehicle_model, daynight_model, NAFNet, light_enhancer,
                           stop_line=None, stop_y=None, tolerance=10,
                           green_zone=None, red_zone=None):
    """Process images from a directory instead of video"""
    if args.vis:
        visualized_dir = "__visualized"
        os.makedirs(visualized_dir, exist_ok=True)

    img_dir = args.img_dir
    output_path = args.output_data
    
    # Initialize trajectory storage
    trajectories = {}
    vehicle_data = defaultdict(list)
    
    # Initialize violation tracking
    violation_count = 0
    violations = {}
    
    # Discover image files
    image_files = [f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    image_files.sort()  # Process in sorted order
    
    # Process each image
    results = []
    frame_count = 0
    
    for img_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        # Resize if needed
        if max(image.shape[0], image.shape[1]) > args.max_dimension:
            scale_factor = args.max_dimension / max(image.shape[0], image.shape[1])
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(image.shape[0] * scale_factor)
            image = cv2.resize(image, (new_width, new_height))
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Set default zones if not provided
        h, w = image.shape[:2]
        if not green_zone and not red_zone and not stop_line and stop_y is None:
            stop_y = h // 2
            green_zone = [(0, stop_y+10), (w, stop_y+10), (w, h), (0, h)]
            red_zone = [(0, 0), (w, 0), (w, stop_y-10), (0, stop_y-10)]
        
        # Process the frame (assuming always red phase for simplicity)
        try:
            processed_frame, frame_detections, new_violations = process_frame(
                image, frame_count, vehicle_model, daynight_model, 
                NAFNet, light_enhancer, device, enhancement_device,
                args.conf, args.iou, trajectories, vehicle_data, 30,  # Assuming 30 fps
                args.memory_efficient, args.disable_enhancement, args.max_dimension,
                args.trajectory_length, args.tracker,
                stop_line, stop_y, tolerance, green_zone, red_zone,
                True,  # Always red phase for simplicity
                violations, violation_count, args.detection_method, 
                None, args.violation_cooldown, args.min_frames_in_green,
                args.trajectory_interpolation, args.debug
            )
            
            # Update violation count
            violation_count += new_violations
            
            # Store results
            for det in frame_detections:
                track_id, cls, x1, y1, x2, y2, conf, speed, is_violation = det
                results.append(f"{img_name} {cls} {x1/image.shape[1]} {y1/image.shape[0]} {(x2-x1)/image.shape[1]} {(y2-y1)/image.shape[0]} {conf} {1 if is_violation else 0}")
            
            # Save visualized image if requested
            if args.vis:
                save_path = os.path.join(visualized_dir, img_name)
                cv2.imwrite(save_path, processed_frame)
        
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
            import traceback
            traceback.print_exc()
        
        frame_count += 1
    
    # Save results to file
    with open(output_path, "w") as f:
        for result in results:
            f.write(result + "\n")
    
    print(f"Total violations detected: {violation_count}")


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    process_video()