# src_main/traffic_light_detector.py

import numpy as np
import cv2

class TrafficLightDetector:
    """
    Detects traffic light state by analyzing pixel colors in a manually defined bounding box using HSV color space
    """
    def __init__(self, bbox=None, 
                 red_threshold=((0, 120, 100), (10, 255, 255), (160, 120, 100), (180, 255, 255)),
                 yellow_threshold=((20, 100, 100), (35, 255, 255)),
                 green_threshold=((40, 40, 50), (95, 255, 255)),
                 min_pixel_percentage=0.1):
        """
        Initialize traffic light detector
        
        Args:
            bbox: Tuple (x1, y1, x2, y2) defining the traffic light bounding box
            red_threshold: HSV thresholds for red detection - two ranges for red wrap-around
                          ((h_min1, s_min1, v_min1), (h_max1, s_max1, v_max1), 
                           (h_min2, s_min2, v_min2), (h_max2, s_max2, v_max2))
            yellow_threshold: HSV threshold for yellow ((h_min, s_min, v_min), (h_max, s_max, v_max))
            green_threshold: HSV threshold for green ((h_min, s_min, v_min), (h_max, s_max, v_max))
            min_pixel_percentage: Minimum percentage of pixels needed to confirm a color
        """
        # CRITICAL: Store original bbox as immutable tuple - NEVER changes during inference
        self.original_bbox = tuple(bbox) if bbox is not None else None
        
        # HSV thresholds
        self.red_threshold = red_threshold
        self.yellow_threshold = yellow_threshold
        self.green_threshold = green_threshold
        self.min_pixel_percentage = min_pixel_percentage
        
        # State tracking
        self.current_state = None
        self.previous_state = None
        self.state_change_frame = None
        
    def set_bbox(self, bbox):
        """Set the traffic light bounding box - only for initialization"""
        if self.original_bbox is None:
            self.original_bbox = tuple(bbox) if bbox is not None else None
        else:
            print("Warning: Attempting to change traffic light bbox during inference - ignored!")
        
    def get_bbox(self):
        """Get the original, unchanging bounding box"""
        return self.original_bbox
        
    def get_state(self):
        """Get the current traffic light state"""
        return self.current_state
        
    def detect(self, frame, frame_count=None):
        """
        Detect traffic light state based on HSV color analysis
        
        Args:
            frame: The video frame (OpenCV format, BGR)
            frame_count: Current frame number (for tracking state changes)
            
        Returns:
            state: 'red', 'yellow', 'green', or None if uncertain
            debug_img: Debug image with annotations
        """
        if self.original_bbox is None:
            return None, frame
            
        # Extract ROI using original coordinates (NEVER changes)
        x1, y1, x2, y2 = [int(coord) for coord in self.original_bbox]
        
        # Ensure coordinates are within frame bounds for safe ROI extraction
        h, w = frame.shape[:2]
        x1_safe = max(0, min(x1, w-1))
        y1_safe = max(0, min(y1, h-1))
        x2_safe = max(x1_safe+1, min(x2, w))
        y2_safe = max(y1_safe+1, min(y2, h))
        
        # Extract ROI
        roi = frame[y1_safe:y2_safe, x1_safe:x2_safe]
        if roi.size == 0:
            return None, frame
            
        # Convert to HSV for color detection
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Red detection (red has two ranges in HSV due to wrap-around)
        red_lower1, red_upper1, red_lower2, red_upper2 = self.red_threshold
        red_mask1 = cv2.inRange(hsv_roi, np.array(red_lower1), np.array(red_upper1))
        red_mask2 = cv2.inRange(hsv_roi, np.array(red_lower2), np.array(red_upper2))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_count = cv2.countNonZero(red_mask)
        
        # Yellow detection
        yellow_lower, yellow_upper = self.yellow_threshold
        yellow_mask = cv2.inRange(hsv_roi, np.array(yellow_lower), np.array(yellow_upper))
        yellow_count = cv2.countNonZero(yellow_mask)
        
        # Green detection
        green_lower, green_upper = self.green_threshold
        green_mask = cv2.inRange(hsv_roi, np.array(green_lower), np.array(green_upper))
        green_count = cv2.countNonZero(green_mask)
        
        # Calculate minimum pixels needed for confident detection
        total_pixels = roi.shape[0] * roi.shape[1]
        min_pixels_needed = int(total_pixels * self.min_pixel_percentage)
        
        # Create debug image
        debug_img = frame.copy()
        
        # Draw stable bounding box using original coordinates
        cv2.rectangle(debug_img, (x1-1, y1-1), (x2+1, y2+1), (0, 0, 0), 3)  # Black outline
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 255, 255), 2)    # White inner
        
        # Determine state based on pixel counts
        max_count = max(red_count, yellow_count, green_count)
        
        if max_count < min_pixels_needed:
            state = None
            text_color = (255, 255, 255)
            state_text = "Unknown"
        elif red_count == max_count:
            state = 'red'
            text_color = (0, 0, 255)
            state_text = "RED"
        elif yellow_count == max_count:
            state = 'yellow'
            text_color = (0, 255, 255)
            state_text = "YELLOW"
        elif green_count == max_count:
            state = 'green'
            text_color = (0, 255, 0)
            state_text = "GREEN"
        
        # Draw state text with outline for visibility
        cv2.putText(debug_img, state_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(debug_img, state_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
            
        # Add pixel counts for debugging
        counts_text = f"R:{red_count} Y:{yellow_count} G:{green_count}"
        cv2.putText(debug_img, counts_text, (x1, y2+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(debug_img, counts_text, (x1, y2+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                   
        # Display bbox coordinates to verify stability
        bbox_text = f"TL BBox (Fixed): {self.original_bbox}"
        cv2.putText(debug_img, bbox_text, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(debug_img, bbox_text, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Track state changes
        if frame_count is not None and state != self.previous_state:
            self.state_change_frame = frame_count
            
        self.previous_state = self.current_state
        self.current_state = state
            
        return state, debug_img
        
    def is_red_phase(self):
        """Check if traffic light is in red phase"""
        return self.current_state == 'red'
        
    def is_yellow_phase(self):
        """Check if traffic light is in yellow phase"""
        return self.current_state == 'yellow'
        
    def is_green_phase(self):
        """Check if traffic light is in green phase"""
        return self.current_state == 'green'
        
    def get_debug_info(self):
        """Get detailed debug information about the detection"""
        if self.original_bbox is None:
            return "No bounding box defined"
            
        return f"Fixed BBox: {self.original_bbox}, Current state: {self.current_state}, " \
               f"HSV Detection, Min pixel %: {self.min_pixel_percentage}"