# utils/visualization.py

import cv2
import numpy as np

class Visualizer:
    """
    Handles visualization of detections, trajectories, zones, and violations
    """
    def __init__(self, config=None):
        """
        Initialize visualizer with configuration
        
        Args:
            config: Dictionary with visualization parameters
        """
        self.config = config or {}
        
        # Define default colors for different vehicle classes
        self.colors = {
            0: (0, 255, 0),    # Green
            1: (255, 0, 0),    # Blue
            2: (0, 0, 255),    # Red
            3: (255, 255, 0),  # Cyan
        }
        
        # Set default parameters if not provided
        if 'trajectory_length' not in self.config:
            self.config['trajectory_length'] = 30
            
    def draw_frame(self, frame, detections, tracker, violations=None, 
              stop_line=None, stop_y=None, green_zone=None, red_zone=None,
              violation_count=0, is_red_phase=False, frame_count=0,
              traffic_light=None):
        """
        Create visualization with all elements including enhanced traffic light info
        """
        # Create a copy for drawing
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Draw zones first (under everything else)
        result = self._draw_zones(result, green_zone, red_zone)
        
        # Draw stop line
        result = self._draw_stop_line(result, stop_line, stop_y, is_red_phase, w, traffic_light)
        
        # CRITICAL: Draw traffic light bbox FIRST to ensure it's always visible
        result = self._draw_traffic_light_bbox(result, traffic_light)
        
        # Draw traffic light status indicator (updated)
        result = self._draw_light_indicator(result, is_red_phase, w, traffic_light)
        
        # Draw violation counter
        result = self._draw_violation_counter(result, violation_count)
        
        # Draw active vehicle counter
        active_vehicles = len(tracker.get_active_tracks(frame_count))
        cv2.putText(result, f"Vehicles: {active_vehicles}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add traffic light state info to left side if available
        if traffic_light and traffic_light.get_state():
            light_state = traffic_light.get_state()
            state_color = (0, 0, 255) if light_state == 'red' else \
                        (0, 255, 255) if light_state == 'yellow' else \
                        (0, 255, 0) if light_state == 'green' else \
                        (255, 255, 255)
            
            cv2.putText(result, f"Light: {light_state.upper()}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(result, f"Light: {light_state.upper()}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 1, cv2.LINE_AA)
        
        # Draw trajectories
        result = self._draw_trajectories(result, tracker, frame_count, 
                                        self.config.get('trajectory_length', 30),
                                        violations, w, h)
        
        # Draw current detections
        result = self._draw_detections(result, detections, violations)
        
        return result
    
    def _draw_zones(self, image, green_zone, red_zone):
        """Draw green and red zones with transparency"""
        if green_zone is None and red_zone is None:
            return image
            
        # Create an overlay for the zones
        overlay = image.copy()
        
        if red_zone:
            # Convert to numpy array for drawing
            pts = np.array(red_zone, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 0, 255))  # Red with alpha
        
        if green_zone:
            # Convert to numpy array for drawing
            pts = np.array(green_zone, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 255, 0))  # Green with alpha
        
        # Blend overlay with original image
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        return image
        
    def _draw_stop_line(self, image, stop_line, stop_y, is_red_phase, width, traffic_light=None):
        """Draw stop line based on configuration"""
        # Determine the stop line color based on the actual traffic light state if available
        light_state = None
        if traffic_light and traffic_light.get_state():
            light_state = traffic_light.get_state()
            
        if light_state == 'red':
            line_color = (0, 0, 255)  # Red
        elif light_state == 'yellow':
            line_color = (0, 255, 255)  # Yellow
        elif light_state == 'green':
            line_color = (0, 255, 0)  # Green
        else:
            # Fall back to simple red/green based on is_red_phase
            line_color = (0, 0, 255) if is_red_phase else (0, 255, 0)  # Red or Green
            
        # Draw the stop line
        if stop_line is not None:
            x1, y1, x2, y2 = stop_line
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), line_color, 3)
        
        elif stop_y is not None:
            # Draw horizontal stop line
            cv2.line(image, (0, stop_y), (width, stop_y), line_color, 3)
            
        return image
        
    def _draw_light_indicator(self, image, is_red_phase, width, traffic_light=None):
        """Draw traffic light status indicator with enhanced information"""
        
        # Get actual traffic light state if available
        if traffic_light and traffic_light.get_state():
            light_state = traffic_light.get_state()
            state_str = light_state.upper() + " LIGHT"
            
            # Set colors based on actual state - always use the detected color
            if light_state == 'red':
                status_color = (0, 0, 255)  # Red
            elif light_state == 'yellow':
                status_color = (0, 255, 255)  # Yellow
            elif light_state == 'green':
                status_color = (0, 255, 0)  # Green
            else:
                status_color = (128, 128, 128)  # Gray for unknown
                state_str = "UNKNOWN"
        else:
            # Fallback to simple red/green based on is_red_phase
            status_color = (0, 0, 255) if is_red_phase else (0, 255, 0)
            state_str = "RED LIGHT" if is_red_phase else "GREEN LIGHT"
        
        # Draw main indicator
        cv2.rectangle(image, (width-180, 10), (width-10, 50), status_color, -1)
        
        # Add text with better visibility (black outline + white text)
        cv2.putText(image, state_str, (width-170, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, state_str, (width-170, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add additional info if traffic light detector is active
        if traffic_light and traffic_light.get_bbox():
            # Show detection confidence or additional debug info
            debug_text = f"TL Detection: Active"
            cv2.putText(image, debug_text, (width-170, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, debug_text, (width-170, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        return image
        
    def _draw_violation_counter(self, image, violation_count):
        """Draw violation counter"""
        # Add violation counter with prominent display
        cv2.rectangle(image, (5, 25), (220, 60), (0, 0, 0), -1)  # Black background
        cv2.putText(image, f"VIOLATIONS: {violation_count}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return image
        
    def _draw_trajectories(self, image, tracker, frame_count, max_length, 
                          violations, width, height):
        """Draw vehicle trajectories"""
        # Get active tracks
        active_track_ids = tracker.get_active_tracks(frame_count)
        
        for track_id in active_track_ids:
            # Get trajectory history for this track
            history = tracker.get_trajectory(track_id, max_length)
            if len(history) < 2:
                continue
                
            # Check if this track has a violation
            has_violated = violations is not None and track_id in violations
            
            # Get color for this track - red for violators
            if has_violated:
                color = (0, 0, 255)  # Red for violators
            else:
                # Get the vehicle class for this track
                track_data = tracker.trajectories.get(track_id, {})
                label = track_data.get("label", 0)
                color = self.colors.get(int(label), (0, 255, 0))
            
            # Draw the trajectory
            for i in range(1, len(history)):
                # Convert center points to integers
                pt1 = (int(history[i-1][0]), int(history[i-1][1]))
                pt2 = (int(history[i][0]), int(history[i][1]))
                
                # Ensure points are within image bounds
                if (0 <= pt1[0] < width and 0 <= pt1[1] < height and 
                    0 <= pt2[0] < width and 0 <= pt2[1] < height):
                    # Make violator trajectories thicker
                    thickness = 3 if has_violated else 2
                    cv2.line(image, pt1, pt2, color, thickness)
            
            # Mark violation point for violators
            if has_violated and violations.get(track_id) and "crossing_point" in violations[track_id]:
                cross_point = violations[track_id]["crossing_point"]
                if cross_point:
                    cross_x, cross_y = cross_point
                    cv2.drawMarker(image, (int(cross_x), int(cross_y)), (0, 0, 255),
                                  markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
        
        return image
        
    def _draw_detections(self, image, detections, violations):
        """Draw bounding boxes and labels for current detections"""
        for det in detections:
            box = det["box"]
            label = det["label"]
            track_id = det["track_id"]
            
            x1, y1, x2, y2 = box.astype(int)
            
            # Check if this vehicle has violated
            has_violated = violations is not None and track_id in violations
            
            # Use different color for violators
            if has_violated:
                color = (0, 0, 255)  # Red for violators
                # Draw a thicker box for violators
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                # Add "VIOLATOR" label
                cv2.putText(image, "VIOLATION", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                color = self.colors.get(int(label), (0, 255, 0))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Simplified label with just track ID and class
                cv2.putText(image, f"ID:{track_id}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        return image
    
    def _draw_traffic_light_bbox(self, image, traffic_light):
        """Draw traffic light bounding box if available - FIXED VERSION"""
        # CRITICAL: Use get_bbox() which returns original_bbox (never changes)
        if traffic_light and traffic_light.get_bbox() is not None:
            bbox = traffic_light.get_bbox()  # This is original_bbox - immutable!
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Draw bounding box with a distinctive color (magenta) and thick lines
            box_color = (255, 0, 255)  # Magenta
            cv2.rectangle(image, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 0), 4)  # Black outline
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 3)  # Magenta box
            
            # Show label based on current state
            if traffic_light.current_state:
                state_str = traffic_light.current_state.upper()
                if traffic_light.current_state == 'red':
                    state_color = (0, 0, 255)
                elif traffic_light.current_state == 'yellow':
                    state_color = (0, 255, 255)
                elif traffic_light.current_state == 'green':
                    state_color = (0, 255, 0)
                else:
                    state_color = (255, 255, 255)
                    
                # Draw text with outline for better visibility
                cv2.putText(image, f"TRAFFIC LIGHT: {state_str}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(image, f"TRAFFIC LIGHT: {state_str}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1, cv2.LINE_AA)
            
            # CRITICAL: Show that bbox is stable
            bbox_debug = f"BBox: {bbox} (FIXED)"
            cv2.putText(image, bbox_debug, (x1, y2+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, bbox_debug, (x1, y2+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            
        return image