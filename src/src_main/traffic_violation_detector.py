# src_main/traffic_violation_detector.py

import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
from tqdm import tqdm
import json

# Import our modular components
from models.detector import VehicleDetector
from models.tracker import VehicleTracker
from utils.visualization import Visualizer
from violations.factory import ViolationDetectorFactory
from traffic_light_detector import TrafficLightDetector

# Import VLM addon
from vlm_addon import SimpleVLMProcessor

# Import enhancement models from original code (if needed)
# from LoLi_IEA.LoLi_IEA import LoLi_IEA
# from basicsr.models import create_model
# from basicsr.utils import img2tensor as _img2tensor
# from basicsr.utils import tensor2img
# from basicsr.utils.options import parse


class TrafficViolationSystem:
    """
    Main class that coordinates all components of the traffic violation detection system
    """
    def __init__(self, args):
        """Initialize with command line arguments"""
        self.args = args
        self.device = self._setup_device()
        
        # Initialize components
        self.vehicle_detector = self._setup_vehicle_detector()
        self.tracker = self._setup_tracker()
        self.traffic_light = self._setup_traffic_light()
        self.violation_detectors = self._setup_violation_detectors()
        self.visualizer = Visualizer(config={
            'trajectory_length': args.trajectory_length
        })
        
        # Enhancement models (commented out for simplicity)
        self.enhancement_models = None
        if not args.disable_enhancement:
            self.enhancement_models = self._setup_enhancement_models()
            
        # Set up zones and stop line
        self.stop_line, self.stop_y, self.green_zone, self.red_zone = self._setup_zones()
        
        # Violation tracking
        self.violations = {}
        self.violation_count = 0
        self.violation_classes = self._parse_violation_classes()
        
        # Traffic light bounding box setup
        self.traffic_light_bbox = self._parse_traffic_light_bbox()
        if self.traffic_light_bbox:
            self.traffic_light.set_bbox(self.traffic_light_bbox)
            print(f"Traffic light detection enabled with bbox: {self.traffic_light_bbox}")
        else:
            print("Traffic light detection disabled - no bounding box provided")
        
        # Determine output directory for VLM images
        vlm_output_dir = "violation_images"
        if hasattr(args, 'output_video') and args.output_video:
            output_dir = os.path.dirname(args.output_video)
            vlm_output_dir = os.path.join(output_dir, "violation_images")

        # Initialize VLM processor (runs in background, doesn't affect performance)
        self.vlm_processor = SimpleVLMProcessor(
            api_key=getattr(args, 'vlm_api_key', None) or os.getenv("GEMINI_API_KEY"),
            batch_size=getattr(args, 'vlm_batch_size', 1),
            enabled=getattr(args, 'enable_vlm', True),
            output_dir=vlm_output_dir
        )
        
    def _setup_device(self):
        """Configure computing device based on arguments"""
        if self.args.memory_efficient:
            return torch.device("cpu")
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    def _setup_vehicle_detector(self):
        """Initialize vehicle detector"""
        return VehicleDetector(
            model_path=self.args.vehicle_model_path,
            conf_threshold=self.args.conf,
            iou_threshold=self.args.iou,
            device=self.device,
            # Pass all arguments to the detector for more flexible setup
            args=self.args
        )
        
    def _setup_tracker(self):
        """Initialize vehicle tracker"""
        return VehicleTracker(
            max_trajectory_length=50,
            speed_estimation_params={
                'fps': 30,  # Default FPS, will be updated for video sources
                'pixels_per_meter': 10,
                'max_speed': 150
            }
        )
        
    def _setup_traffic_light(self):
        """Initialize traffic light detector with HSV parameters only"""
        # Parse HSV thresholds
        red_threshold = self._parse_hsv_threshold(self.args.red_threshold_hsv, is_red=True)
        yellow_threshold = self._parse_hsv_threshold(self.args.yellow_threshold_hsv)
        green_threshold = self._parse_hsv_threshold(self.args.green_threshold_hsv)
        
        return TrafficLightDetector(
            bbox=None,  # Will be set later
            red_threshold=red_threshold,
            yellow_threshold=yellow_threshold,
            green_threshold=green_threshold,
            min_pixel_percentage=self.args.min_pixel_percentage
        )

    def _parse_hsv_threshold(self, threshold_str, is_red=False):
        """Parse HSV threshold string to appropriate format"""
        try:
            values = list(map(int, threshold_str.split(',')))
            if is_red:
                # Red has two ranges in HSV due to wrap-around (0-10 and 160-180)
                if len(values) != 12:
                    print(f"Warning: Red HSV threshold should have 12 values, got {len(values)}. Using defaults.")
                    return ((0, 120, 100), (10, 255, 255), (160, 120, 100), (180, 255, 255))
                return ((values[0], values[1], values[2]), 
                        (values[3], values[4], values[5]),
                        (values[6], values[7], values[8]),
                        (values[9], values[10], values[11]))
            else:
                # Yellow and green have one range
                if len(values) != 6:
                    print(f"Warning: HSV threshold should have 6 values, got {len(values)}. Using defaults.")
                    if threshold_str == self.args.yellow_threshold_hsv:
                        return ((20, 100, 100), (35, 255, 255))
                    else:  # green
                        return ((40, 40, 50), (95, 255, 255))
                return ((values[0], values[1], values[2]), 
                        (values[3], values[4], values[5]))
        except Exception as e:
            print(f"Error parsing HSV threshold: {threshold_str}")
            print(f"Exception: {e}")
            # Return sensible defaults
            if is_red:
                return ((0, 120, 100), (10, 255, 255), (160, 120, 100), (180, 255, 255))
            elif threshold_str == self.args.yellow_threshold_hsv:
                return ((20, 100, 100), (35, 255, 255))
            else:  # green
                return ((40, 40, 50), (95, 255, 255))

    def _parse_traffic_light_bbox(self):
        """Parse traffic light bounding box if provided"""
        if not hasattr(self.args, 'traffic_light_bbox') or not self.args.traffic_light_bbox:
            return None
            
        try:
            bbox = tuple(map(float, self.args.traffic_light_bbox.split(',')))
            if len(bbox) != 4:
                print("Invalid traffic light bbox format. Must be x1,y1,x2,y2")
                return None
            print(f"Traffic light bbox set at: {bbox}")
            return bbox
        except:
            print(f"Error parsing traffic light bbox: {self.args.traffic_light_bbox}")
            return None
        
    def _setup_violation_detectors(self):
        """Initialize violation detectors based on detection method"""
        # Get the parsed stop line, not the raw string
        stop_line, stop_y, green_zone, red_zone = self._setup_zones()
        
        # Ensure the stop_line is in the correct format (x1, y1, x2, y2)
        if stop_line and len(stop_line) > 4:
            print(f"Warning: Stop line has {len(stop_line)} values, expected 4. Using first 4 values.")
            stop_line = stop_line[:4]
        
        config = {
            'stop_line': stop_line,
            'stop_y': stop_y,  # Use the parsed stop_y, not args.stop_y
            'tolerance': self.args.tolerance,
            'green_zone': green_zone,
            'red_zone': red_zone,
            'min_frames_in_green': self.args.min_frames_in_green,
            'interpolation_steps': self.args.trajectory_interpolation
        }
        
        return ViolationDetectorFactory.create(
            self.args.detection_method, 
            config
        )
        
    def _setup_enhancement_models(self):
        """Initialize image enhancement models"""
        # This is simplified - in a real implementation, we would
        # initialize NAFNet and LoLi_IEA models here
        return None
        
    def _setup_zones(self):
        """Setup stop line and zones for violation detection"""
        # Parse stop line parameters
        stop_line = None
        stop_y = self.args.stop_y
        
        if self.args.stop_line:
            try:
                stop_line = tuple(map(float, self.args.stop_line.split(',')))
                if len(stop_line) != 4:
                    print("Invalid stop line format. Must be x1,y1,x2,y2")
                    stop_line = None
                else:
                    print(f"Stop line set at: {stop_line}")
            except:
                print(f"Error parsing stop line: {self.args.stop_line}")
                stop_line = None
        
        # Parse green and red zones (same as before)
        green_zone = None
        red_zone = None
        
        if self.args.green_zone:
            try:
                coords = list(map(float, self.args.green_zone.split(',')))
                if len(coords) < 6 or len(coords) % 2 != 0:
                    print("Invalid green zone format. Must have at least 3 points (6 coordinates)")
                else:
                    green_zone = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                    print(f"Green zone set with {len(green_zone)} points")
            except:
                print(f"Error parsing green zone: {self.args.green_zone}")
        
        if self.args.red_zone:
            try:
                coords = list(map(float, self.args.red_zone.split(',')))
                if len(coords) < 6 or len(coords) % 2 != 0:
                    print("Invalid red zone format. Must have at least 3 points (6 coordinates)")
                else:
                    red_zone = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                    print(f"Red zone set with {len(red_zone)} points")
            except:
                print(f"Error parsing red zone: {self.args.red_zone}")
                
        return stop_line, stop_y, green_zone, red_zone
        
    def _parse_violation_classes(self):
        """Parse the violation classes to monitor"""
        if self.args.violation_classes == "all":
            return []
            
        try:
            classes = list(map(int, self.args.violation_classes.split(',')))
            print(f"Monitoring classes for violations: {classes}")
            return classes
        except:
            print(f"Error parsing violation classes: {self.args.violation_classes}. Monitoring all classes.")
            return []
            
    def process_video(self):
        """Process a video file or camera stream"""
        # Set up video capture
        try:
            # Check if input is a camera index
            if self.args.input.isdigit():
                cap = cv2.VideoCapture(int(self.args.input))
            else:
                cap = cv2.VideoCapture(self.args.input)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video source: {self.args.input}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Update tracker with actual FPS
            self.tracker.speed_estimation_params['fps'] = fps
            
            # Set up default zones if not provided
            self._setup_default_zones_if_needed(width, height)
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.args.output_video, fourcc, fps, (width, height))
            
            # Initialize data file
            with open(self.args.output_data, 'w') as f:
                f.write("frame,track_id,class,x1,y1,x2,y2,confidence,speed_kmh,violation\n")
            
            # Process video frames
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.args.input.isdigit() is False else float('inf')
            
            # Set up visualization (if not in headless mode)
            show_visualization = self.args.vis and not self.args.headless
            display_available = self._check_display_available() if show_visualization else False
            
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
                if self.args.skip_frames > 0 and frame_count % (self.args.skip_frames + 1) != 0:
                    frame_count += 1
                    pbar.update(1)
                    continue
                
                # Resize if needed
                frame = self._resize_frame_if_needed(frame)
                
                # Process the current frame
                try:
                    processed_frame, frame_detections, new_violations = self.process_frame(frame, frame_count)
                    
                    # Update violation count
                    self.violation_count += new_violations
                    
                    # Write results to output file
                    self._write_detection_results(frame_count, frame_detections)
                    
                    # Write frame to output video
                    out.write(processed_frame)
                    
                    # Save preview frames if requested
                    self._save_preview_frame_if_needed(processed_frame, frame_count)
                    
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
            
            print(f"Total violations detected: {self.violation_count}")
            
            # Export VLM results
            if hasattr(self, 'vlm_processor') and self.vlm_processor.enabled:
                print("📄 Exporting VLM results...")
                self.vlm_processor.export_report("violations_with_plates.json")
                self.vlm_processor.stop()
            
        except Exception as e:
            print(f"Error processing video: {e}")
            import traceback
            traceback.print_exc()
            
    def process_frame(self, frame, frame_count):
        """
        Process a single video frame with HSV traffic light detection
        
        Args:
            frame: The video frame to process
            frame_count: Current frame count
            
        Returns:
            processed_frame: Frame with visualizations
            detections: List of detection data for output
            new_violations: Number of new violations in this frame
        """
        # Track new violations in this frame
        new_violations = 0
        
        # Step 1: Detect traffic light state (if bbox defined)
        if self.traffic_light_bbox:
            light_state, frame = self.traffic_light.detect(frame, frame_count)
            # Only consider red light as a violation phase
            # Yellow and green are valid traffic states and should not trigger violations
            is_red_phase = light_state == 'red'
                
        else:
            # Fallback: use default red light assumption or other logic
            is_red_phase = True  # Default to always monitoring violations
            
        # Step 2: Detect and track vehicles
        detection_results = self.vehicle_detector.detect_and_track(
            frame, 
            tracker_type=self.args.tracker,
            persist=True
        )
        
        # Step 3: Update tracker with new detections
        detections = self.tracker.update(detection_results, frame_count)
        
        # Step 4: Check for violations (only during red phase)
        if is_red_phase:
            # For each tracked vehicle, check for violations
            for det in detections:
                track_id = det["track_id"]
                
                # Skip if this vehicle already has a violation
                if track_id in self.violations:
                    # Check cooldown period
                    if frame_count - self.violations[track_id]["frame"] < self.args.violation_cooldown:
                        continue
                        
                # Check if this class should be monitored
                is_monitored = len(self.violation_classes) == 0 or int(det["label"]) in self.violation_classes
                if not is_monitored:
                    continue
                    
                # Check for violations using all configured detectors
                for detector in self.violation_detectors:
                    violated, violation_info = detector.check_violation(
                        self.tracker, track_id, frame_count, is_red_phase
                    )
                    
                    if violated:
                        new_violations += 1
                        self.violations[track_id] = violation_info
                        if self.args.debug:
                            print(f"🚨 VIOLATION DETECTED! Vehicle {track_id}, frame {frame_count}, " 
                                f"type: {violation_info.get('violation_type', 'unknown')}")
                        
                        # Add to VLM processing (non-blocking, runs in background)
                        # Convert bbox to list format [x1, y1, x2, y2]
                        bbox = det["box"]
                        if hasattr(bbox, 'tolist'):  # If it's numpy array
                            bbox = bbox.tolist()
                        elif not isinstance(bbox, list):  # If it's tuple or other
                            bbox = list(bbox)
                            
                        self.vlm_processor.add_violation(
                            frame=frame,
                            bbox=bbox,
                            track_id=track_id,
                            frame_count=frame_count,
                            violation_info=violation_info
                        )
                        break
        
        # Step 5: Create formatted detection data for output
        frame_detections = []
        for det in detections:
            box = det["box"]
            track_id = det["track_id"]
            label = det["label"]
            score = det["score"]
            speed = det["speed"]
            
            # Check if this is a violation
            is_violation = track_id in self.violations
            
            # Add to frame detections for output
            frame_detections.append((
                int(track_id), int(label), 
                int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                score, speed, is_violation
            ))
        
        # Step 6: Create visualization
        result_frame = self.visualizer.draw_frame(
            frame, detections, self.tracker, 
            self.violations, self.stop_line, self.stop_y, 
            self.green_zone, self.red_zone,
            self.violation_count + new_violations, 
            is_red_phase, frame_count,
            self.traffic_light
        )
        
        return result_frame, frame_detections, new_violations
    
    def _check_simulated_red_light(self, frame_count):
        """Check if current frame is in a simulated red light phase"""
        # Always red if red_light flag is set
        if self.args.red_light:
            return True
            
        # Check against red light ranges if provided
        if hasattr(self.args, 'red_light_ranges') and self.args.red_light_ranges:
            for start, end in self.args.red_light_ranges:
                if start <= frame_count <= end:
                    return True
                    
        return False
        
    def _setup_default_zones_if_needed(self, width, height):
        """Set up default zones and stop line if not provided"""
        if not self.stop_line and self.stop_y is None and not self.red_zone and not self.green_zone:
            print("No zones or stop line provided. Using default setup.")
            self.stop_y = height // 2
            
            # Create default zones based on stop line
            self.green_zone = [(0, self.stop_y+10), (width, self.stop_y+10), (width, height), (0, height)]
            self.red_zone = [(0, 0), (width, 0), (width, self.stop_y-10), (0, self.stop_y-10)]
            
            print(f"Default stop line at y={self.stop_y}")
            print(f"Default green zone (below stop line): {self.green_zone}")
            print(f"Default red zone (above stop line): {self.red_zone}")
            
    def _check_display_available(self):
        """Check if display is available for visualization"""
        try:
            # Test if we can create a window
            cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("Test Window")
            return True
        except:
            print("Warning: Could not create display window. Running in headless mode.")
            return False
            
    def _resize_frame_if_needed(self, frame):
        """Resize frame if needed based on arguments"""
        # Check if process_resolution is specified
        if hasattr(self.args, 'process_resolution') and self.args.process_resolution:
            try:
                target_width, target_height = map(int, self.args.process_resolution.split('x'))
                frame = cv2.resize(frame, (target_width, target_height))
                return frame
            except:
                print(f"Invalid resolution format: {self.args.process_resolution}")
                
        # Check max dimension
        if max(frame.shape[0], frame.shape[1]) > self.args.max_dimension:
            scale_factor = self.args.max_dimension / max(frame.shape[0], frame.shape[1])
            new_width = int(frame.shape[1] * scale_factor)
            new_height = int(frame.shape[0] * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height))
            
        return frame
        
    def _write_detection_results(self, frame_count, detections):
        """Write detection results to output file"""
        with open(self.args.output_data, 'a') as f:
            for det in detections:
                track_id, cls, x1, y1, x2, y2, conf, speed, is_violation = det
                f.write(f"{frame_count},{track_id},{cls},{x1},{y1},{x2},{y2},{conf},{speed},{1 if is_violation else 0}\n")
                
        # Output real-time data for streaming (if enabled)
        if hasattr(self.args, 'realtime_output') and self.args.realtime_output:
            current_violations = len([det for det in detections if det[8]])  # Count violations in this frame
            current_detections = len(detections)
            
            realtime_data = {
                'frame': frame_count,
                'detections': current_detections,
                'violations': self.violation_count,
                'current_violations': current_violations,
                'timestamp': time.time()
            }
            
            # Print JSON data that the backend can parse
            print(f"REALTIME_DATA: {json.dumps(realtime_data)}")
            sys.stdout.flush()  # Ensure immediate output
                
    def _save_preview_frame_if_needed(self, frame, frame_count):
        """Save preview frames at intervals if enabled"""
        if self.args.save_preview_frames > 0 and frame_count % self.args.save_preview_frames == 0:
            # Use the output folder structure that backend expects
            if hasattr(self.args, 'output_video') and self.args.output_video:
                # Extract the output directory from output_video path
                output_dir = os.path.dirname(self.args.output_video)
                preview_dir = os.path.join(output_dir, "preview_frames")
            else:
                # Fallback to local preview_frames directory
                preview_dir = "preview_frames"
            
            os.makedirs(preview_dir, exist_ok=True)
            preview_path = os.path.join(preview_dir, f"frame_{frame_count:06d}.jpg")
            
            # Save with higher quality for better streaming
            cv2.imwrite(preview_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Also save to a 'latest' frame for easier access
            latest_path = os.path.join(preview_dir, "latest_frame.jpg")
            cv2.imwrite(latest_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # if self.args.debug:
            #     print(f"Preview frame saved: {preview_path}")
                
            # Output debug info about frame saving
            if hasattr(self.args, 'realtime_output') and self.args.realtime_output:
                print(f"FRAME_SAVED: {preview_path}")
                sys.stdout.flush()
                
    def process_image_directory(self):
        """Process images from a directory instead of video"""
        if not self.args.img_dir:
            print("No image directory specified")
            return
            
        img_dir = self.args.img_dir
        vis_dir = "__visualized" if self.args.vis else None
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
            
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
            image = self._resize_frame_if_needed(image)
            
            # Set up default zones if not provided
            h, w = image.shape[:2]
            self._setup_default_zones_if_needed(w, h)
            
            # Process the frame (always using red phase for simplicity)
            try:
                processed_frame, frame_detections, new_violations = self.process_frame(image, frame_count)
                
                # Update violation count
                self.violation_count += new_violations
                
                # Store results
                for det in frame_detections:
                    track_id, cls, x1, y1, x2, y2, conf, speed, is_violation = det
                    results.append(f"{img_name},{track_id},{cls},{x1},{y1},{x2},{y2},{conf},{speed},{1 if is_violation else 0}")
                
                # Save visualized image if requested
                if vis_dir:
                    save_path = os.path.join(vis_dir, img_name)
                    cv2.imwrite(save_path, processed_frame)
                    
            except Exception as e:
                print(f"Error processing image {img_name}: {e}")
                import traceback
                traceback.print_exc()
                
            frame_count += 1
            
        # Save results to file
        with open(self.args.output_data, "w") as f:
            f.write("file,track_id,class,x1,y1,x2,y2,confidence,speed,violation\n")
            for result in results:
                f.write(result + "\n")
                
        print(f"Total violations detected: {self.violation_count}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Traffic Violation Detection")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to video file or camera index (e.g. 0 for webcam)")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="Optional: Process images from directory instead of video")
    
    parser.add_argument("--conf", type=float, default=0.65)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--vehicle_model_path", type=str, default="/home/quangsang/Study/maiAnhEm/soict-hackathon-2024/weights/vehicle/epoch_best.pt")
    parser.add_argument("--daynight_model_path", type=str, default="/home/quangsang/Study/maiAnhEm/soict-hackathon-2024/weights/day_night/best.pt")
    parser.add_argument("--loli_iea_model_dir", type=str, default="/home/quangsang/Study/maiAnhEm/soict-hackathon-2024/weights/LoLi_IEA")

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
    parser.add_argument("--tracker_config", type=str, default="None",
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
    
    # Traffic light detection - HSV ONLY
    parser.add_argument("--traffic_light_bbox", type=str, default=None,
                        help="Format: x1,y1,x2,y2 - Bounding box for traffic light detection")
    
    # HSV thresholds for traffic light (ONLY HSV SUPPORTED)
    parser.add_argument("--red_threshold_hsv", type=str, default="0,120,100,10,255,255,160,120,100,180,255,255",
                        help="HSV threshold for red light detection (h_min1,s_min1,v_min1,h_max1,s_max1,v_max1,h_min2,s_min2,v_min2,h_max2,s_max2,v_max2)")
    parser.add_argument("--yellow_threshold_hsv", type=str, default="20,100,100,35,255,255",
                        help="HSV threshold for yellow light detection (h_min,s_min,v_min,h_max,s_max,v_max)")
    parser.add_argument("--green_threshold_hsv", type=str, default="40,40,50,95,255,255",
                        help="HSV threshold for green light detection (h_min,s_min,v_min,h_max,s_max,v_max)")
    
    parser.add_argument("--min_pixel_percentage", type=float, default=0.1,
                        help="Minimum percentage of pixels needed to confirm a color")
    
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
    parser.add_argument("--use_tensorrt", action="store_true", help="Convert model to TensorRT for inference acceleration")
    parser.add_argument("--half_precision", action="store_true", help="Use FP16 for TensorRT (further acceleration)")
    parser.add_argument("--tensorrt_workspace", type=int, default=8, help="Workspace limit (GB) for TensorRT")
    parser.add_argument("--tensorrt_dynamic", action="store_true", help="Use dynamic batch size for TensorRT")
    
    # Real-time output for streaming
    parser.add_argument("--realtime_output", action="store_true", help="Enable real-time data output for streaming")
    
    # VLM parameters
    parser.add_argument("--enable_vlm", action="store_true", help="Enable license plate extraction using VLM")
    parser.add_argument("--vlm_api_key", type=str, help="Gemini API key for VLM (or use GEMINI_API_KEY env var)")
    parser.add_argument("--vlm_batch_size", type=int, default=1, help="VLM batch size for processing")
    
    return parser.parse_args()


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Set GPU memory growth
    if torch.cuda.is_available():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Create traffic violation system
    system = TrafficViolationSystem(args)
    
    # Process video or images
    if args.img_dir:
        system.process_image_directory()
    else:
        system.process_video()
    
    print("Processing complete!")


if __name__ == "__main__":
    main()