# src_main/traffic_light_evaluator.py 

"""
Improved Traffic Light Evaluator

A standalone tool to evaluate traffic light detection with interactive bbox selection.
Input: Video file and optional traffic light bounding box
Output: Video with traffic light detection visualization and state logs
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Import the TrafficLightDetector class
from traffic_light_detector import TrafficLightDetector

# Global variables for bbox selection
bbox_selection = False
drawing = False
ix, iy = -1, -1
x, y, w, h = 0, 0, 0, 0
roi_selected = False
bbox_coords = None

def mouse_callback(event, x_pos, y_pos, flags, param):
    global ix, iy, drawing, x, y, w, h, roi_selected, bbox_coords
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x_pos, y_pos
        x, y, w, h = 0, 0, 0, 0
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            x = min(x_pos, ix)
            y = min(y_pos, iy)
            w = abs(x_pos - ix)
            h = abs(y_pos - iy)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x = min(x_pos, ix)
        y = min(y_pos, iy)
        w = abs(x_pos - ix)
        h = abs(y_pos - iy)
        
        if w > 5 and h > 5:  # Minimum size to consider a valid selection
            roi_selected = True
            bbox_coords = (x, y, x+w, y+h)
            print(f"Selected bounding box: {bbox_coords}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Improved Traffic Light Detection Evaluation Tool")
    
    # Input/output arguments
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to input video file")
    parser.add_argument("--output", type=str, default="traffic_light_eval.mp4",
                        help="Path to output video file")
    parser.add_argument("--log", type=str, default="traffic_light_states.csv",
                        help="Path to output CSV log file")
    
    # Traffic light detection parameters
    parser.add_argument("--traffic_light_bbox", type=str, default=None,
                        help="Format: x1,y1,x2,y2 - Bounding box for traffic light detection. If not provided, interactive selection will be used.")
    parser.add_argument("--use_hsv", action="store_true", default=True,
                        help="Use HSV color space instead of RGB for better color detection")
    
    # RGB thresholds
    parser.add_argument("--red_threshold_rgb", type=str, default="150,60,60",
                        help="RGB threshold for red light detection")
    parser.add_argument("--yellow_threshold_rgb", type=str, default="150,150,50",
                        help="RGB threshold for yellow light detection")
    parser.add_argument("--green_threshold_rgb", type=str, default="60,150,60",
                        help="RGB threshold for green light detection")
    
    # HSV thresholds
    parser.add_argument("--red_threshold_hsv", type=str, default="0,120,100,10,255,255,160,120,100,180,255,255",
                        help="HSV threshold for red light detection (h_min1,s_min1,v_min1,h_max1,s_max1,v_max1,h_min2,s_min2,v_min2,h_max2,s_max2,v_max2)")
    parser.add_argument("--yellow_threshold_hsv", type=str, default="20,100,100,35,255,255",
                        help="HSV threshold for yellow light detection (h_min,s_min,v_min,h_max,s_max,v_max)")
    parser.add_argument("--green_threshold_hsv", type=str, default="40,40,50,95,255,255",
                        help="HSV threshold for green light detection (h_min,s_min,v_min,h_max,s_max,v_max)")
    
    parser.add_argument("--min_pixel_percentage", type=float, default=0.1,
                        help="Minimum percentage of pixels needed to confirm a color")
    
    # Visualization/processing parameters
    parser.add_argument("--vis", action="store_true",
                        help="Show visualization during processing")
    parser.add_argument("--skip_frames", type=int, default=0,
                        help="Skip N frames between each processed frame")
    parser.add_argument("--process_resolution", type=str, default=None,
                        help="Process at this resolution, format: WIDTHxHEIGHT")
    parser.add_argument("--save_preview_frames", type=int, default=0,
                        help="Save preview frames at intervals (0 to disable)")
    parser.add_argument("--debug_view", action="store_true", 
                        help="Include detailed debug visualization in output")
    parser.add_argument("--save_roi", action="store_true", default=True,
                        help="Save the ROI images of traffic light for manual inspection")
    
    return parser.parse_args()


def parse_bbox(bbox_str):
    """Parse bbox string to tuple of coordinates"""
    try:
        return tuple(map(float, bbox_str.split(',')))
    except Exception as e:
        print(f"Error parsing traffic light bbox: {bbox_str}")
        print(f"Exception: {e}")
        return None


def parse_threshold(threshold_str):
    """Parse RGB threshold string to tuple of integers"""
    try:
        return tuple(map(int, threshold_str.split(',')))
    except Exception as e:
        print(f"Error parsing threshold: {threshold_str}")
        print(f"Exception: {e}")
        return None


def parse_hsv_threshold(threshold_str, is_red=False):
    """Parse HSV threshold string to appropriate format"""
    try:
        values = list(map(int, threshold_str.split(',')))
        if is_red:
            # Red has two ranges in HSV
            if len(values) != 12:
                raise ValueError("Red HSV threshold should have 12 values")
            return ((values[0], values[1], values[2]), 
                    (values[3], values[4], values[5]),
                    (values[6], values[7], values[8]),
                    (values[9], values[10], values[11]))
        else:
            # Yellow and green have one range
            if len(values) != 6:
                raise ValueError("Yellow/Green HSV threshold should have 6 values")
            return ((values[0], values[1], values[2]), 
                    (values[3], values[4], values[5]))
    except Exception as e:
        print(f"Error parsing HSV threshold: {threshold_str}")
        print(f"Exception: {e}")
        return None


def select_bbox_interactively(video_path):
    """
    Allow user to select a bounding box interactively from the video
    
    Returns:
        Tuple (x1, y1, x2, y2) of the selected bounding box
    """
    global bbox_selection, roi_selected, bbox_coords
    
    # Reset globals
    bbox_selection = True
    roi_selected = False
    bbox_coords = None
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None
    
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Could not read the first frame")
        cap.release()
        return None
    
    # Create window and set mouse callback
    window_name = "Select Traffic Light Bounding Box (drag to select, press Enter when done)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n=== INTERACTIVE BOUNDING BOX SELECTION ===")
    print("1. Click and drag to select the traffic light")
    print("2. Press Enter to confirm selection or 'n' for next frame")
    print("3. Press 'q' to quit\n")
    
    frame_count = 0
    
    while True:
        # Make a copy of the frame to draw on
        img_display = frame.copy()
        
        # If we're drawing or have selected a ROI, display it
        if drawing or roi_selected:
            cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if roi_selected:
                # Show the selected region enlarged
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    # Resize ROI for better visibility (3x)
                    display_h, display_w = 120, 120
                    roi_display = cv2.resize(roi, (display_w, display_h))
                    
                    # Position in top-right corner
                    img_h, img_w = img_display.shape[:2]
                    top_right_y, top_right_x = 10, img_w - display_w - 10
                    
                    # Create a background rectangle
                    cv2.rectangle(img_display, 
                                 (top_right_x-5, top_right_y-5),
                                 (top_right_x+display_w+5, top_right_y+display_h+5),
                                 (0, 0, 0), -1)
                    
                    # Overlay ROI
                    img_display[top_right_y:top_right_y+display_h, 
                               top_right_x:top_right_x+display_w] = roi_display
                    
                    # Add text
                    cv2.putText(img_display, f"Selected ROI", 
                               (top_right_x, top_right_y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show current coordinates on screen
        if roi_selected:
            coord_text = f"Selected: ({x}, {y}, {x+w}, {y+h})"
        else:
            coord_text = "No selection yet"
            
        cv2.putText(img_display, coord_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img_display, coord_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add frame counter
        cv2.putText(img_display, f"Frame: {frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img_display, f"Frame: {frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display the image
        cv2.imshow(window_name, img_display)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Handle key press
        if key == ord('q'):
            # Quit
            bbox_coords = None
            break
        elif key == 13:  # Enter key
            # Confirm selection
            if roi_selected:
                break
        elif key == ord('n'):
            # Next frame
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                # Loop back to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                frame_count = 0
            else:
                frame_count += 1
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    return bbox_coords


def evaluate_traffic_light(args):
    """Run traffic light detection evaluation"""
    print(f"Evaluating traffic light detection on: {args.input}")
    
    # Parse or interactively select traffic light bbox
    traffic_light_bbox = None
    if args.traffic_light_bbox:
        traffic_light_bbox = parse_bbox(args.traffic_light_bbox)
        print(f"Using provided bounding box: {traffic_light_bbox}")
    else:
        print("No bounding box provided. Starting interactive selection...")
        traffic_light_bbox = select_bbox_interactively(args.input)
        
    if not traffic_light_bbox:
        print("No valid traffic light bounding box! Exiting.")
        return
    
    # Parse threshold values
    red_threshold_rgb = parse_threshold(args.red_threshold_rgb)
    yellow_threshold_rgb = parse_threshold(args.yellow_threshold_rgb)
    green_threshold_rgb = parse_threshold(args.green_threshold_rgb)
    
    red_threshold_hsv = parse_hsv_threshold(args.red_threshold_hsv, is_red=True)
    yellow_threshold_hsv = parse_hsv_threshold(args.yellow_threshold_hsv)
    green_threshold_hsv = parse_hsv_threshold(args.green_threshold_hsv)
    
    # Create traffic light detector
    traffic_light = TrafficLightDetector(
        bbox=traffic_light_bbox,
        use_hsv=args.use_hsv,
        red_threshold_rgb=red_threshold_rgb,
        yellow_threshold_rgb=yellow_threshold_rgb,
        green_threshold_rgb=green_threshold_rgb,
        red_threshold_hsv=red_threshold_hsv,
        yellow_threshold_hsv=yellow_threshold_hsv,
        green_threshold_hsv=green_threshold_hsv,
        min_pixel_percentage=args.min_pixel_percentage
    )
    
    # Parse process resolution
    target_width, target_height = None, None
    if args.process_resolution:
        try:
            target_width, target_height = map(int, args.process_resolution.split('x'))
            print(f"Processing at resolution: {target_width}x{target_height}")
        except:
            print(f"Invalid resolution format: {args.process_resolution}. Using original resolution.")
    
    # Open video capture
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Could not open video: {args.input}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Override with target resolution if specified
    if target_width and target_height:
        # Adjust bounding box for new resolution
        x1, y1, x2, y2 = traffic_light_bbox
        x1 = int(x1 * (target_width / width))
        x2 = int(x2 * (target_width / width))
        y1 = int(y1 * (target_height / height))
        y2 = int(y2 * (target_height / height))
        traffic_light_bbox = (x1, y1, x2, y2)
        traffic_light.set_bbox(traffic_light_bbox)
        
        # Update dimensions
        width, height = target_width, target_height
    
    # Create output directories
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create preview directory
    preview_dir = None
    if args.save_preview_frames > 0:
        preview_dir = os.path.join(os.path.dirname(args.output), "preview_frames")
        os.makedirs(preview_dir, exist_ok=True)
        print(f"Will save preview frames to {preview_dir}/")
    
    # Create ROI directory if enabled
    roi_dir = None
    if args.save_roi:
        roi_dir = os.path.join(os.path.dirname(args.output), "traffic_light_roi")
        os.makedirs(roi_dir, exist_ok=True)
        print(f"Will save ROI frames to {roi_dir}/")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Initialize state log file
    with open(args.log, 'w') as f:
        f.write("frame,timestamp,state,red_count,yellow_count,green_count,total_pixels,min_pixels_needed\n")
    
    # Process video
    frame_count = 0
    state_history = []
    color_history = {'red': [], 'yellow': [], 'green': [], 'unknown': []}
    
    # Progress bar
    pbar = tqdm(total=total_frames, desc="Processing video")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames if requested
        if args.skip_frames > 0 and frame_count % (args.skip_frames + 1) != 0:
            frame_count += 1
            pbar.update(1)
            continue
        
        # Resize if needed
        if target_width and target_height:
            frame = cv2.resize(frame, (target_width, target_height))
        
        # Process frame with traffic light detector
        state, debug_image = traffic_light.detect(frame, frame_count)
        
        # Calculate time in seconds
        time_sec = frame_count / fps
        
        # Create timestamp
        mins = int(time_sec // 60)
        secs = int(time_sec % 60)
        timestamp = f"{mins:02d}:{secs:02d}"
        
        # Add frame count and timestamp
        y_pos = 30
        # Add black outline for better visibility
        cv2.putText(debug_image, f"Frame: {frame_count}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(debug_image, f"Frame: {frame_count}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        y_pos += 30
        # Add black outline for better visibility
        cv2.putText(debug_image, f"Time: {timestamp}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(debug_image, f"Time: {timestamp}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add current state with color
        y_pos += 30
        state_color = (255, 255, 255)
        state_text = "UNKNOWN"
        
        if state == 'red':
            state_color = (0, 0, 255)
            state_text = "RED"
        elif state == 'yellow':
            state_color = (0, 255, 255)
            state_text = "YELLOW"
        elif state == 'green':
            state_color = (0, 255, 0)
            state_text = "GREEN"
        
        # Add black outline for better visibility
        cv2.putText(debug_image, f"State: {state_text}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(debug_image, f"State: {state_text}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 1, cv2.LINE_AA)
        
        # Extract color counts for logging
        x1, y1, x2, y2 = [int(coord) for coord in traffic_light_bbox]
        roi = frame[y1:y2, x1:x2]
        
        if roi.size > 0:
            if args.use_hsv:
                # HSV processing
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # Red detection (red has two ranges in HSV)
                red_lower1, red_upper1, red_lower2, red_upper2 = red_threshold_hsv
                red_mask1 = cv2.inRange(hsv_roi, np.array(red_lower1), np.array(red_upper1))
                red_mask2 = cv2.inRange(hsv_roi, np.array(red_lower2), np.array(red_upper2))
                red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                red_count = cv2.countNonZero(red_mask)
                
                # Yellow detection
                yellow_lower, yellow_upper = yellow_threshold_hsv
                yellow_mask = cv2.inRange(hsv_roi, np.array(yellow_lower), np.array(yellow_upper))
                yellow_count = cv2.countNonZero(yellow_mask)
                
                # Green detection
                green_lower, green_upper = green_threshold_hsv
                green_mask = cv2.inRange(hsv_roi, np.array(green_lower), np.array(green_upper))
                green_count = cv2.countNonZero(green_mask)
                
                total_pixels = roi.shape[0] * roi.shape[1]
            else:
                # RGB processing
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                
                # Red detection (R high, G low, B low)
                red_mask = ((rgb_roi[:,:,0] > red_threshold_rgb[0]) & 
                          (rgb_roi[:,:,1] < red_threshold_rgb[1]) & 
                          (rgb_roi[:,:,2] < red_threshold_rgb[2]))
                red_count = np.sum(red_mask)
                
                # Yellow detection (R high, G high, B low)
                yellow_mask = ((rgb_roi[:,:,0] > yellow_threshold_rgb[0]) & 
                             (rgb_roi[:,:,1] > yellow_threshold_rgb[1]) & 
                             (rgb_roi[:,:,2] < yellow_threshold_rgb[2]))
                yellow_count = np.sum(yellow_mask)
                
                # Green detection (R low, G high, B low)
                green_mask = ((rgb_roi[:,:,0] < green_threshold_rgb[0]) & 
                            (rgb_roi[:,:,1] > green_threshold_rgb[1]) & 
                            (rgb_roi[:,:,2] < green_threshold_rgb[2]))
                green_count = np.sum(green_mask)
                
                total_pixels = red_mask.size
            
            min_pixels_needed = int(total_pixels * args.min_pixel_percentage)
            
            # Save ROI if enabled
            if roi_dir and frame_count % 30 == 0:  # Save every 30 frames
                roi_path = os.path.join(roi_dir, f"roi_frame_{frame_count:06d}.jpg")
                cv2.imwrite(roi_path, roi)
                
                # Save color masks as well
                if args.use_hsv:
                    # Create visualization masks
                    red_viz = np.zeros_like(roi)
                    yellow_viz = np.zeros_like(roi)
                    green_viz = np.zeros_like(roi)
                    
                    red_viz[red_mask1 > 0] = [0, 0, 255]
                    red_viz[red_mask2 > 0] = [0, 0, 255]
                    yellow_viz[yellow_mask > 0] = [0, 255, 255]
                    green_viz[green_mask > 0] = [0, 255, 0]
                    
                    cv2.imwrite(os.path.join(roi_dir, f"red_mask_{frame_count:06d}.jpg"), red_viz)
                    cv2.imwrite(os.path.join(roi_dir, f"yellow_mask_{frame_count:06d}.jpg"), yellow_viz)
                    cv2.imwrite(os.path.join(roi_dir, f"green_mask_{frame_count:06d}.jpg"), green_viz)
            
            # Log state
            with open(args.log, 'a') as f:
                f.write(f"{frame_count},{timestamp},{state if state else 'unknown'},{red_count},{yellow_count},{green_count},{total_pixels},{min_pixels_needed}\n")
        
        # Add state to history
        state_history.append(state)
        
        # Update color history
        if state == 'red':
            color_history['red'].append(frame_count)
        elif state == 'yellow':
            color_history['yellow'].append(frame_count)
        elif state == 'green':
            color_history['green'].append(frame_count)
        else:
            color_history['unknown'].append(frame_count)
        
        # Create debug view with pixel visualization if requested
        if args.debug_view and roi.size > 0:
            # Create masked images for visualization
            if args.use_hsv:
                # Already have the masks from HSV processing
                red_vis = np.zeros_like(roi)
                red_vis[red_mask > 0] = [0, 0, 255]
                
                yellow_vis = np.zeros_like(roi)
                yellow_vis[yellow_mask > 0] = [0, 255, 255]
                
                green_vis = np.zeros_like(roi)
                green_vis[green_mask > 0] = [0, 255, 0]
            else:
                # Create visualization from RGB masks
                red_vis = np.zeros_like(roi)
                red_vis[red_mask] = [0, 0, 255]
                
                yellow_vis = np.zeros_like(roi)
                yellow_vis[yellow_mask] = [0, 255, 255]
                
                green_vis = np.zeros_like(roi)
                green_vis[green_mask] = [0, 255, 0]
            
            # Resize for better visibility
            h, w = roi.shape[:2]
            scale = 3
            roi_resized = cv2.resize(roi, (w*scale, h*scale))
            red_vis_resized = cv2.resize(red_vis, (w*scale, h*scale))
            yellow_vis_resized = cv2.resize(yellow_vis, (w*scale, h*scale))
            green_vis_resized = cv2.resize(green_vis, (w*scale, h*scale))
            
            # Create a composite image
            margin = 5
            vis_width = w*scale
            vis_height = h*scale
            
            # Calculate layout
            debug_vis_height = vis_height*4 + margin*5
            debug_vis_width = vis_width + margin*2
            
            debug_vis = np.zeros((debug_vis_height, debug_vis_width, 3), dtype=np.uint8)
            
            # Add original ROI
            y_offset = margin
            debug_vis[y_offset:y_offset+vis_height, margin:margin+vis_width] = roi_resized
            # Add black outline for better visibility
            cv2.putText(debug_vis, "Original", (margin, y_offset-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(debug_vis, "Original", (margin, y_offset-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Add red mask visualization
            y_offset += vis_height + margin
            debug_vis[y_offset:y_offset+vis_height, margin:margin+vis_width] = red_vis_resized
            # Add black outline for better visibility
            cv2.putText(debug_vis, f"Red ({red_count}/{total_pixels})", (margin, y_offset-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(debug_vis, f"Red ({red_count}/{total_pixels})", (margin, y_offset-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
            # Add yellow mask visualization
            y_offset += vis_height + margin
            debug_vis[y_offset:y_offset+vis_height, margin:margin+vis_width] = yellow_vis_resized
            # Add black outline for better visibility
            cv2.putText(debug_vis, f"Yellow ({yellow_count}/{total_pixels})", (margin, y_offset-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(debug_vis, f"Yellow ({yellow_count}/{total_pixels})", (margin, y_offset-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            
            # Add green mask visualization
            y_offset += vis_height + margin
            debug_vis[y_offset:y_offset+vis_height, margin:margin+vis_width] = green_vis_resized
            # Add black outline for better visibility
            cv2.putText(debug_vis, f"Green ({green_count}/{total_pixels})", (margin, y_offset-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(debug_vis, f"Green ({green_count}/{total_pixels})", (margin, y_offset-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            # Add debug visualization to the main image (resize if needed)
            main_img_height, main_img_width = debug_image.shape[:2]
            if debug_vis_height > main_img_height // 3:
                scale_factor = (main_img_height // 3) / debug_vis_height
                debug_vis_width = int(debug_vis_width * scale_factor)
                debug_vis_height = int(debug_vis_height * scale_factor)
                debug_vis = cv2.resize(debug_vis, (debug_vis_width, debug_vis_height))
            
            # Place debug visualization in top-right corner
            padding = 10
            roi_x = main_img_width - debug_vis_width - padding
            roi_y = padding
            
            # Create a background rectangle for better visibility
            cv2.rectangle(debug_image, 
                         (roi_x-5, roi_y-5),
                         (roi_x+debug_vis_width+5, roi_y+debug_vis_height+5),
                         (0, 0, 0), -1)
            
            # Create overlay region
            debug_image[roi_y:roi_y+debug_vis_height, roi_x:roi_x+debug_vis_width] = debug_vis
        
        # Write frame to output video
        out.write(debug_image)
        
        # Save preview frame if requested
        if preview_dir and args.save_preview_frames > 0 and frame_count % args.save_preview_frames == 0:
            preview_path = os.path.join(preview_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(preview_path, debug_image)
        
        # Display if visualization enabled
        if args.vis:
            cv2.imshow('Traffic Light Detection', debug_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                # Pause/play on 'p' key
                cv2.waitKey(0)
        
        frame_count += 1
        pbar.update(1)
    
    # Clean up
    cap.release()
    out.release()
    if args.vis:
        cv2.destroyAllWindows()
    
    pbar.close()
    
    # Report statistics
    total_frames_processed = len(state_history)
    if total_frames_processed > 0:
        red_frames = state_history.count('red')
        yellow_frames = state_history.count('yellow')
        green_frames = state_history.count('green')
        unknown_frames = state_history.count(None)
        
        print("\n=== Traffic Light Detection Statistics ===")
        print(f"Total frames processed: {total_frames_processed}")
        print(f"Red frames: {red_frames} ({red_frames/total_frames_processed*100:.1f}%)")
        print(f"Yellow frames: {yellow_frames} ({yellow_frames/total_frames_processed*100:.1f}%)")
        print(f"Green frames: {green_frames} ({green_frames/total_frames_processed*100:.1f}%)")
        print(f"Unknown frames: {unknown_frames} ({unknown_frames/total_frames_processed*100:.1f}%)")
        
        # Calculate transitions
        transitions = []
        prev_state = None
        for i, state in enumerate(state_history):
            if state != prev_state and i > 0:
                transitions.append((prev_state, state, i))
            prev_state = state
            
        print(f"\nDetected {len(transitions)} state transitions:")
        for prev, curr, frame in transitions[:10]:  # Show first 10 transitions
            prev_str = prev if prev else "unknown"
            curr_str = curr if curr else "unknown"
            print(f"Frame {frame}: {prev_str} -> {curr_str}")
            
        if len(transitions) > 10:
            print(f"... and {len(transitions) - 10} more transitions")
            
        # Make recommendations if many unknown frames
        if unknown_frames > total_frames_processed * 0.5:
            print("\n=== RECOMMENDATIONS ===")
            print("High number of unknown states detected. Consider:")
            print("1. Adjusting the bounding box to better capture the traffic light")
            print("2. Decreasing min_pixel_percentage (currently: {args.min_pixel_percentage})")
            print("3. Using HSV color space instead of RGB for better color detection")
            print("4. Adjusting color thresholds to match your specific traffic light")
            print("5. Check the saved ROIs to see what the actual traffic light region looks like")
    
    print(f"\nResults saved to:")
    print(f"- Video: {args.output}")
    print(f"- Log: {args.log}")
    if preview_dir:
        print(f"- Preview frames: {preview_dir}/")
    if roi_dir:
        print(f"- ROI images: {roi_dir}/")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Run evaluation
    evaluate_traffic_light(args)