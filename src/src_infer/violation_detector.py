"""
Refactored Traffic Violation Detector

This module provides a clean, organized approach to traffic violation detection
with proper dimension handling and backend synchronization.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class ViolationDetector:
    """
    A class to handle traffic violation detection with better organization and dimension management.
    """
    
    def __init__(self, args):
        self.args = args
        self.trajectories = {}
        self.violations = {}
        self.violation_count = 0
        
        # Store original and processing dimensions for consistent coordinate handling
        self.original_width = 0
        self.original_height = 0
        self.processing_width = 0
        self.processing_height = 0
        self.scale_factor = 1.0
        
        # Parse zone coordinates
        self.green_zone = self._parse_zone_coords(args.green_zone)
        self.red_zone = self._parse_zone_coords(args.red_zone)
        self.stop_line = self._parse_stop_line(args.stop_line)
        self.stop_y = args.stop_y
        
        # Parse violation classes
        self.violation_classes = self._parse_violation_classes()
        
        # Parse red light phases
        self.red_light_ranges = self._parse_red_light_ranges()
        
        print("ViolationDetector initialized with:")
        print(f"- Detection method: {args.detection_method}")
        print(f"- Green zone: {len(self.green_zone) if self.green_zone else 0} points")
        print(f"- Red zone: {len(self.red_zone) if self.red_zone else 0} points")
        print(f"- Stop line: {self.stop_line}")
        print(f"- Violation classes: {self.violation_classes if self.violation_classes else 'all'}")
    
    def _parse_zone_coords(self, zone_str: Optional[str]) -> Optional[List[Tuple[float, float]]]:
        """Parse zone coordinates from string format"""
        if not zone_str:
            return None
        
        try:
            coords = list(map(float, zone_str.split(',')))
            if len(coords) < 6 or len(coords) % 2 != 0:
                print(f"Invalid zone format: {zone_str}. Must have at least 3 points (6 coordinates)")
                return None
            
            # Convert to list of (x, y) points
            zone = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            return zone
        except Exception as e:
            print(f"Error parsing zone coordinates: {e}")
            return None
    
    def _parse_stop_line(self, stop_line_str: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
        """Parse stop line coordinates from string format"""
        if not stop_line_str:
            return None
        
        try:
            coords = tuple(map(float, stop_line_str.split(',')))
            if len(coords) != 4:
                print("Invalid stop line format. Must be x1,y1,x2,y2")
                return None
            return coords
        except Exception as e:
            print(f"Error parsing stop line: {e}")
            return None
    
    def _parse_violation_classes(self) -> List[int]:
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
    
    def _parse_red_light_ranges(self) -> List[Tuple[int, int]]:
        """Parse red light frame ranges"""
        if not self.args.red_light_frames:
            return []
        
        try:
            frames = list(map(int, self.args.red_light_frames.split(',')))
            if len(frames) % 2 != 0:
                print("Invalid red light frames format. Must be start1,end1,start2,end2,...")
                return []
            
            ranges = [(frames[i], frames[i+1]) for i in range(0, len(frames), 2)]
            print(f"Red light phases: {ranges}")
            return ranges
        except Exception as e:
            print(f"Error parsing red light frames: {e}")
            return []
    
    def set_video_dimensions(self, original_width: int, original_height: int, 
                           processing_width: int, processing_height: int):
        """Set video dimensions for coordinate scaling"""
        self.original_width = original_width
        self.original_height = original_height
        self.processing_width = processing_width
        self.processing_height = processing_height
        self.scale_factor = processing_width / original_width
        
        print(f"Video dimensions set:")
        print(f"- Original: {original_width}x{original_height}")
        print(f"- Processing: {processing_width}x{processing_height}")
        print(f"- Scale factor: {self.scale_factor:.3f}")
        
        # Scale zone coordinates if they were provided in original dimensions
        if self.green_zone:
            self.green_zone = self._scale_zone_coords(self.green_zone)
        if self.red_zone:
            self.red_zone = self._scale_zone_coords(self.red_zone)
        if self.stop_line:
            self.stop_line = self._scale_stop_line(self.stop_line)
        if self.stop_y:
            self.stop_y = int(self.stop_y * self.scale_factor)
    
    def _scale_zone_coords(self, zone: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Scale zone coordinates to processing dimensions"""
        return [(x * self.scale_factor, y * self.scale_factor) for x, y in zone]
    
    def _scale_stop_line(self, stop_line: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Scale stop line coordinates to processing dimensions"""
        x1, y1, x2, y2 = stop_line
        return (x1 * self.scale_factor, y1 * self.scale_factor, 
                x2 * self.scale_factor, y2 * self.scale_factor)
    
    def setup_default_zones(self, width: int, height: int):
        """Set up default zones if none provided"""
        if (not self.stop_line and not self.stop_y and 
            not self.red_zone and not self.green_zone):
            
            print("No zones or stop line provided. Using default setup.")
            self.stop_y = height // 2
            
            # Create default zones based on stop line
            self.green_zone = [(0, self.stop_y+10), (width, self.stop_y+10), (width, height), (0, height)]
            self.red_zone = [(0, 0), (width, 0), (width, self.stop_y-10), (0, self.stop_y-10)]
            
            print(f"Default stop line at y={self.stop_y}")
            print(f"Default green zone: {self.green_zone}")
            print(f"Default red zone: {self.red_zone}")
    
    def is_red_light_phase(self, frame_count: int) -> bool:
        """Determine if the current frame is in a red light phase"""
        if self.args.red_light:
            return True
        
        for start, end in self.red_light_ranges:
            if start <= frame_count <= end:
                return True
        
        return False
    
    def point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm"""
        if not polygon:
            return False
        
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
    
    def check_stop_line_violation(self, prev_pos: Tuple[float, float], 
                                curr_pos: Tuple[float, float]) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """Check if a vehicle crossed the stop line"""
        if self.stop_line is not None:
            # Line intersection logic
            movement_line = (prev_pos[0], prev_pos[1], curr_pos[0], curr_pos[1])
            crossed, cross_point = self._line_intersection(movement_line, self.stop_line)
            return crossed, cross_point
        
        elif self.stop_y is not None:
            # Simple horizontal stop line
            prev_x, prev_y = prev_pos
            curr_x, curr_y = curr_pos
            
            # Check if trajectory crosses the stop line
            if ((prev_y < self.stop_y - self.args.tolerance and curr_y > self.stop_y + self.args.tolerance) or
                (prev_y > self.stop_y + self.args.tolerance and curr_y < self.stop_y - self.args.tolerance)):
                
                # Calculate approximate crossing point
                if prev_y != curr_y:
                    t = (self.stop_y - prev_y) / (curr_y - prev_y)
                    cross_x = prev_x + t * (curr_x - prev_x)
                    return True, (cross_x, self.stop_y)
                else:
                    return True, (prev_x, self.stop_y)
        
        return False, None
    
    def _line_intersection(self, line1: Tuple[float, float, float, float], 
                          line2: Tuple[float, float, float, float]) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """Determine if two line segments intersect"""
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
            margin = 1e-9
            return (min(a, b) - margin <= c <= max(a, b) + margin)
        
        if (is_between(line1[0], line1[2], x) and 
            is_between(line1[1], line1[3], y) and 
            is_between(line2[0], line2[2], x) and 
            is_between(line2[1], line2[3], y)):
            return True, (x, y)
        
        return False, None
    
    def check_zone_transition(self, track: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Check if a vehicle moved from green zone to red zone"""
        if not self.red_zone or not self.green_zone:
            return False, {}
        
        if "history" not in track or len(track["history"]) < 2:
            return False, {}
        
        current_pos = track["history"][-1]
        in_red_zone = self.point_in_polygon(current_pos, self.red_zone)
        
        # Initialize zone tracking if not present
        if "zone_history" not in track:
            track["zone_history"] = []
            track["frames_in_green"] = 0
            track["zone_violated"] = False
        
        # Get current zone
        current_zone = None
        if self.point_in_polygon(current_pos, self.green_zone):
            current_zone = "green"
            track["frames_in_green"] += 1
        elif self.point_in_polygon(current_pos, self.red_zone):
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
            if track["frames_in_green"] >= self.args.min_frames_in_green:
                if self.args.debug:
                    print(f"Track {track.get('track_id', 0)} was in green zone for {track['frames_in_green']} frames, now in red zone")
                
                track["zone_violated"] = True
                violation_info = {
                    "track_id": track.get("track_id", 0),
                    "violation_type": "zone_transition",
                    "frames_in_green": track["frames_in_green"]
                }
                return True, violation_info
        
        return False, {}
    
    def check_violation_with_interpolation(self, track: Dict[str, Any], 
                                         frame_count: int) -> Tuple[bool, Dict[str, Any]]:
        """Advanced violation detection with interpolation"""
        if "history" not in track or len(track["history"]) < 2:
            return False, {}
        
        prev_pos = track["history"][-2]
        curr_pos = track["history"][-1]
        
        # Create interpolated trajectory
        interpolated_points = self._interpolate_trajectory(prev_pos, curr_pos)
        
        # Initialize violation flags
        stop_line_violated = False
        zone_violated = False
        crossing_point = None
        
        # Method 1: Check stop line violation
        if self.args.detection_method in ["stop_line", "combined"]:
            for i in range(1, len(interpolated_points)):
                p1 = interpolated_points[i-1]
                p2 = interpolated_points[i]
                
                crossed, cross_point = self.check_stop_line_violation(p1, p2)
                if crossed:
                    stop_line_violated = True
                    crossing_point = cross_point
                    break
        
        # Method 2: Check zone transition
        if self.args.detection_method in ["zone", "combined"]:
            zone_result, zone_info = self.check_zone_transition(track)
            if zone_result:
                zone_violated = True
        
        # Determine overall violation
        violation_occurred = False
        violation_info = {}
        
        if self.args.detection_method == "stop_line":
            violation_occurred = stop_line_violated
            if violation_occurred:
                violation_info = {
                    "track_id": track.get("track_id", 0),
                    "violation_type": "stop_line",
                    "crossing_point": crossing_point
                }
        elif self.args.detection_method == "zone":
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
    
    def _interpolate_trajectory(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                              steps: Optional[int] = None) -> List[Tuple[float, float]]:
        """Create interpolated points between two trajectory points"""
        if steps is None:
            steps = self.args.trajectory_interpolation
        
        points = []
        for i in range(steps):
            t = i / (steps - 1)
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            points.append((x, y))
        return points
    
    def process_detections(self, detections: List[Dict[str, Any]], 
                         frame_count: int, fps: float) -> int:
        """Process detections and check for violations"""
        new_violations = 0
        is_red_phase = self.is_red_light_phase(frame_count)
        
        for detection in detections:
            track_id = detection.get('track_id')
            if track_id is None:
                continue
            
            # Check if this class should be monitored
            label = detection.get('label', 0)
            is_monitored = len(self.violation_classes) == 0 or int(label) in self.violation_classes
            
            # Update trajectory
            center_x = (detection['box'][0] + detection['box'][2]) / 2
            center_y = (detection['box'][1] + detection['box'][3]) / 2
            
            if track_id not in self.trajectories:
                self.trajectories[track_id] = {
                    "history": [(center_x, center_y)],
                    "label": label,
                    "first_seen": frame_count,
                    "last_seen": frame_count,
                    "track_id": track_id
                }
            else:
                # Update trajectory
                self.trajectories[track_id]["history"].append((center_x, center_y))
                self.trajectories[track_id]["last_seen"] = frame_count
                
                # Limit history length
                if len(self.trajectories[track_id]["history"]) > 50:
                    self.trajectories[track_id]["history"] = self.trajectories[track_id]["history"][-50:]
                
                # Check for violation
                if is_red_phase and is_monitored and track_id not in self.violations:
                    violated, violation_info = self.check_violation_with_interpolation(
                        self.trajectories[track_id], frame_count
                    )
                    
                    if violated:
                        violation_type = violation_info.get("violation_type", "unknown")
                        
                        if self.args.debug:
                            print(f"🚨 VIOLATION DETECTED! Vehicle {track_id}, frame {frame_count}, type: {violation_type}")
                        else:
                            print(f"🚨 VIOLATION DETECTED! Vehicle {track_id}, frame {frame_count}")
                        
                        new_violations += 1
                        self.violations[track_id] = {
                            "frame": frame_count,
                            "type": violation_type,
                            "crossing_point": violation_info.get("crossing_point", None)
                        }
        
        self.violation_count += new_violations
        return new_violations


class GeometryUtils:
    """Utility functions for geometric calculations"""
    
    @staticmethod
    def calculate_speed(track: Dict[str, Any], fps: float, pixels_per_meter: float = 10) -> float:
        """Estimate speed in km/h based on trajectory and framerate"""
        if "history" not in track or len(track["history"]) < 2:
            return 0
        
        # Calculate distance in pixels between last two positions
        p1 = track["history"][-2]
        p2 = track["history"][-1]
        distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # Convert to meters
        distance_meters = distance_pixels / pixels_per_meter
        
        # Calculate time between frames in hours
        time_hours = 1 / (fps * 3600)
        
        # Calculate speed in km/h
        speed_kmh = (distance_meters / 1000) / time_hours
        
        return min(speed_kmh, 150)  # Cap at reasonable max speed 