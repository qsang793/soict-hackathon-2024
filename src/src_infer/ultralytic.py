import argparse
import os
import sys
import time
from collections import defaultdict

# Add parent directory to system path to find the LoLi_IEA module
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

# Now the imports should work
from LoLi_IEA.LoLi_IEA import LoLi_IEA
from utils.yolo_utils import visualize_images

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO Video Inference Script")
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
        # If we encounter a memory error, return the original image
        if "CUDA out of memory" in str(e):
            print(f"Warning: Not enough memory for deblurring, returning original image")
            # Convert back to BGR
            return cv2.cvtColor(img * 255, cv2.COLOR_RGB2BGR).astype(np.uint8)
        else:
            # Re-raise if it's not a memory error
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
        # If CUDA out of memory, try on CPU
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
        # Use the track method instead of predict
        results = model.track(
            source=source,
            conf=conf,
            iou=iou,
            verbose=False,
            device=device,
            tracker="bytetrack.yaml",  # Use specified tracker
            persist=persist        # Keep tracking between frames
        )[0]
        return results
    except RuntimeError as e:
        # If CUDA out of memory, try on CPU
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


def visualize_tracked_image(image, detections, trajectories=None, trajectory_length=30):
    """
    Visualize detections with tracking information
    """
    # Define colors for different vehicle classes
    colors = {
        0: (0, 255, 0),    # Green for class 0
        1: (255, 0, 0),    # Blue for class 1
        2: (0, 0, 255),    # Red for class 2
        3: (255, 255, 0),  # Cyan for class 3
        4: (0, 255, 255),  # Yellow for class 4
        5: (255, 0, 255),  # Magenta for class 5
    }
    
    # Draw bounding boxes and labels
    for i, (box, label, score, track_id, speed) in enumerate(detections):
        x1, y1, x2, y2 = box.astype(int)
        color = colors.get(int(label), (0, 255, 0))
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Add label with track ID
        label_text = f"ID:{track_id} Class:{int(label)} {score:.2f} {speed:.1f} km/h"
        cv2.putText(image, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw trajectories if available
    if trajectories:
        for track_id, track in trajectories.items():
            if "history" not in track or len(track["history"]) < 2:
                continue
                
            # Get color for this track
            label = track.get("label", 0)
            color = colors.get(int(label), (0, 255, 0))
            
            # Draw the recent trajectory (limited by trajectory_length)
            history = track["history"][-trajectory_length:] if len(track["history"]) > trajectory_length else track["history"]
            for i in range(1, len(history)):
                # Convert center points to integers
                pt1 = (int(history[i-1][0]), int(history[i-1][1]))
                pt2 = (int(history[i][0]), int(history[i][1]))
                
                # Ensure points are within image bounds
                h, w = image.shape[:2]
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(image, pt1, pt2, color, 2)
    
    return image


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
    
    # Apply smoothing (you could use moving average for better results)
    return min(speed_kmh, 150)  # Cap speed at 150 km/h to filter outliers


def process_video():
    ## Parse arguments ----------------------------------------------
    args = parse_arguments()
    
    # Check if process_resolution is specified
    target_width, target_height = None, None
    if args.process_resolution:
        try:
            target_width, target_height = map(int, args.process_resolution.split('x'))
        except:
            print(f"Invalid resolution format: {args.process_resolution}. Using original resolution.")
    
    output_video_path = args.output_video
    output_data_path = args.output_data
    
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
    
    # Load enhancement models if not disabled
    if not args.disable_enhancement:
        light_enhancer = LoLi_IEA(args.loli_iea_model_dir, enhancement_device)
        
        # Configure NAFNet based on device
        opt_path = "weights/NAFNNet/NAFNet-width64.yml"
        opt = parse(opt_path, is_train=False)
        opt["dist"] = False
        
        # Set device in the options
        if enhancement_device.type == "cpu":
            opt["num_gpus"] = 0  # Force CPU
        else:
            opt["num_gpus"] = 1
            
        # Create model with the appropriate device settings
        NAFNet = create_model(opt)
    else:
        light_enhancer = None
        NAFNet = None
    
    # Initialize trajectory storage
    trajectories = {}  # track_id -> {"history": [(x,y), ...], "label": class_label}
    vehicle_data = defaultdict(list)  # For storing metrics about each vehicle
    
    ## Warmup models -----------------------------------------------
    print("Warming up models...")
    dummy_frame = np.zeros((640, 480, 3), dtype=np.uint8)
    vehicle_model.predict(dummy_frame, device=device, verbose=False)
    daynight_model.predict(dummy_frame, device=device, verbose=False)
    
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
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # Initialize data file
            with open(output_data_path, 'w') as f:
                f.write("frame,track_id,class,x1,y1,x2,y2,confidence,speed_kmh\n")
            
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
                
                # Process frame
                try:
                    processed_frame, frame_detections = process_frame(
                        frame, frame_count, vehicle_model, daynight_model, 
                        NAFNet, light_enhancer, device, enhancement_device,
                        args.conf, args.iou, trajectories, vehicle_data, fps,
                        args.memory_efficient, args.disable_enhancement, args.max_dimension,
                        args.trajectory_length, args.tracker
                    )
                    
                    # Write results to output file
                    with open(output_data_path, 'a') as f:
                        for det in frame_detections:
                            track_id, cls, x1, y1, x2, y2, conf, speed = det
                            f.write(f"{frame_count},{track_id},{cls},{x1},{y1},{x2},{y2},{conf},{speed}\n")
                    
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
            
        except Exception as e:
            print(f"Error processing video: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Process images from directory
        process_image_directory(args, device, enhancement_device, 
                               vehicle_model, daynight_model, NAFNet, light_enhancer)
    
    print(f"Processing complete. Results saved to {output_video_path} and {output_data_path}")


def process_frame(frame, frame_count, vehicle_model, daynight_model, NAFNet, light_enhancer,
                 device, enhancement_device, conf_threshold, iou_threshold, 
                 trajectories, vehicle_data, fps, memory_efficient=False,
                 disable_enhancement=False, max_dimension=1280, trajectory_length=30,
                 tracker_type="bytetrack"):
    """Process a single frame, returning the processed frame and detection data"""
    # Clear CUDA cache to prevent memory issues
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Copy original frame for visualization
    original_frame = frame.copy()
    processed_frame = None
    
    try:
        # Enhance the image if enhancement is enabled
        if not disable_enhancement:
            # Classify as day or night
            day_night_cls = infer_classify(daynight_model, frame, 
                                          "cpu" if memory_efficient else device)
            
            # Apply appropriate enhancement
            if day_night_cls == 0:  # Day
                processed_frame = deblur(NAFNet, frame, device, memory_efficient, max_dimension)
            else:  # Night
                # Try to enhance with light enhancer
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
                    
                    # Create or update trajectory data
                    if track_id not in trajectories:
                        trajectories[track_id] = {
                            "history": [(center_x, center_y)],
                            "label": label,
                            "first_seen": frame_count,
                            "last_seen": frame_count
                        }
                    else:
                        trajectories[track_id]["history"].append((center_x, center_y))
                        trajectories[track_id]["last_seen"] = frame_count
                        
                        # Limit history length
                        if len(trajectories[track_id]["history"]) > 50:
                            trajectories[track_id]["history"] = trajectories[track_id]["history"][-50:]
                    
                    # Calculate speed
                    speed = calculate_speed(trajectories[track_id], fps)
                    
                    # Store detection data
                    vehicle_data[track_id].append({
                        "frame": frame_count,
                        "box": box,
                        "label": label,
                        "score": score,
                        "speed": speed
                    })
                    
                    # Add to frame detections for output
                    frame_detections.append((
                        int(track_id), int(label), 
                        int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                        score, speed
                    ))
                    
                    # Add to visualization list
                    viz_detections.append((
                        box, label, score, int(track_id), speed
                    ))
                
                # Visualize detections with trajectories
                result_frame = visualize_tracked_image(original_frame, viz_detections, trajectories, trajectory_length)
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
                        score, 0.0  # No speed info without tracking
                    ))
                
                result_frame = visualize_tracked_image(original_frame, viz_detections)
        else:
            # No detections
            result_frame = original_frame
            frame_detections = []
    
    except Exception as e:
        print(f"Error processing frame {frame_count}: {e}")
        import traceback
        traceback.print_exc()
        result_frame = original_frame
        frame_detections = []
    
    # Add timestamp and frame number
    cv2.putText(result_frame, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(result_frame, time_str, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Add counter for tracked vehicles
    active_vehicles = len([tid for tid, track in trajectories.items() 
                          if track["last_seen"] >= frame_count - 30])
    cv2.putText(result_frame, f"Vehicles: {active_vehicles}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return result_frame, frame_detections


def process_image_directory(args, device, enhancement_device, 
                           vehicle_model, daynight_model, NAFNet, light_enhancer):
    """Process images from a directory instead of video"""
    if args.vis:
        visualized_dir = "__visualized"
        os.makedirs(visualized_dir, exist_ok=True)

    img_dir = args.img_dir
    output_path = args.output_data
    
    # Initialize trajectory storage
    trajectories = {}
    vehicle_data = defaultdict(list)
    
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
        
        # Process the frame
        try:
            processed_frame, frame_detections = process_frame(
                image, frame_count, vehicle_model, daynight_model, 
                NAFNet, light_enhancer, device, enhancement_device,
                args.conf, args.iou, trajectories, vehicle_data, 30,  # Assuming 30 fps
                args.memory_efficient, args.disable_enhancement, args.max_dimension,
                args.trajectory_length, args.tracker
            )
            
            # Store results
            for det in frame_detections:
                track_id, cls, x1, y1, x2, y2, conf, speed = det
                results.append(f"{img_name} {cls} {x1/image.shape[1]} {y1/image.shape[0]} {(x2-x1)/image.shape[1]} {(y2-y1)/image.shape[0]} {conf}")
            
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


if __name__ == "__main__":
    # Set environment variable for PyTorch memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Run the pipeline
    process_video()