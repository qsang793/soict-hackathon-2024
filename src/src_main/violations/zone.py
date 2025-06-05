# violations/zone.py

from .detector import ViolationDetector
import numpy as np

class ZoneViolationDetector(ViolationDetector):
    """
    Detects zone transition violations (vehicle moving from green to red zone)
    """
    def __init__(self, green_zone=None, red_zone=None, min_frames_in_green=3):
        """
        Initialize zone violation detector
        
        Args:
            green_zone: List of (x,y) points defining the legal approach zone
            red_zone: List of (x,y) points defining the violation zone
            min_frames_in_green: Minimum frames vehicle must be in green zone
        """
        self.green_zone = green_zone
        self.red_zone = red_zone
        self.min_frames_in_green = min_frames_in_green
        self.zone_data = {}  # Track zone data by track_id
        
    def check_violation(self, tracker, track_id, frame_count, is_red_phase):
        """Check if a vehicle has moved from green zone to red zone during red phase"""
        if not is_red_phase:
            return False, {}
            
        # Skip if zones are not defined
        if self.red_zone is None or self.green_zone is None:
            return False, {}
            
        # Get current position
        trajectory = tracker.get_trajectory(track_id)
        if not trajectory:
            return False, {}
            
        current_pos = trajectory[-1]
        
        # Initialize zone tracking if not present
        if track_id not in self.zone_data:
            self.zone_data[track_id] = {
                "zone_history": [],
                "frames_in_green": 0,
                "zone_violated": False
            }
            
        # Get current zone
        in_green = self._point_in_polygon(current_pos, self.green_zone)
        in_red = self._point_in_polygon(current_pos, self.red_zone)
        
        if in_green:
            current_zone = "green"
            self.zone_data[track_id]["frames_in_green"] += 1
        elif in_red:
            current_zone = "red"
        else:
            current_zone = "outside"
            
        # Update zone history
        self.zone_data[track_id]["zone_history"].append(current_zone)
        
        # Keep only the last 50 zone entries
        if len(self.zone_data[track_id]["zone_history"]) > 50:
            self.zone_data[track_id]["zone_history"] = self.zone_data[track_id]["zone_history"][-50:]
            
        # Check for green -> red transition
        if in_red and not self.zone_data[track_id]["zone_violated"]:
            # Check if vehicle was previously in green zone long enough
            if self.zone_data[track_id]["frames_in_green"] >= self.min_frames_in_green:
                self.zone_data[track_id]["zone_violated"] = True
                return True, {
                    "violation_type": "zone_transition",
                    "track_id": track_id,
                    "frame": frame_count,
                    "frames_in_green": self.zone_data[track_id]["frames_in_green"]
                }
                
        return False, {}
        
    def _point_in_polygon(self, point, polygon):
        """
        Check if a point is inside a polygon using ray casting algorithm
        
        Args:
            point: (x, y) tuple
            polygon: [(x1, y1), (x2, y2), ...] list of points
            
        Returns:
            Boolean, True if the point is inside the polygon
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