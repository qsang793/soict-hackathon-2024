# violations/stop_line.py

from .detector import ViolationDetector

class StopLineViolationDetector(ViolationDetector):
    """
    Detects stop line violations
    """
    def __init__(self, stop_line=None, stop_y=None, tolerance=10, interpolation_steps=5):
        """
        Initialize stop line violation detector
        
        Args:
            stop_line: (x1, y1, x2, y2) tuple defining a stop line
            stop_y: Y-coordinate for a horizontal stop line (if stop_line is None)
            tolerance: Tolerance in pixels for the stop line
            interpolation_steps: Number of interpolation steps for trajectory
        """
        self.stop_line = stop_line
        self.stop_y = stop_y
        self.tolerance = tolerance
        self.interpolation_steps = interpolation_steps
        
    def check_violation(self, tracker, track_id, frame_count, is_red_phase):
        """Check if a vehicle has crossed the stop line during red phase"""
        if not is_red_phase:
            return False, {}
            
        # Get interpolated trajectory
        points = tracker.interpolate_trajectory(track_id, self.interpolation_steps)
        if len(points) < 2:
            return False, {}
            
        # Check each pair of consecutive points
        for i in range(1, len(points)):
            p1 = points[i-1]
            p2 = points[i]
            
            # Check if these points cross the stop line
            crossed, cross_point = self._check_line_crossing(p1, p2)
            
            if crossed:
                return True, {
                    "violation_type": "stop_line",
                    "track_id": track_id,
                    "frame": frame_count,
                    "crossing_point": cross_point
                }
                
        return False, {}
        
    def _check_line_crossing(self, p1, p2):
        """
        Check if a trajectory between two points crosses the stop line
        
        Returns:
            (crossed, crossing_point): Tuple with crossing status and point
        """
        if self.stop_line is not None:
            # Define the line segment from previous to current position
            movement_line = (p1[0], p1[1], p2[0], p2[1])
            
            # Check if the segments intersect
            crossed, cross_point = self._line_intersection(movement_line, self.stop_line)
            return crossed, cross_point
            
        elif self.stop_y is not None:
            # Simple horizontal stop line
            prev_x, prev_y = p1
            curr_x, curr_y = p2
            
            # Check if trajectory crosses the stop line
            if ((prev_y < self.stop_y - self.tolerance and curr_y > self.stop_y + self.tolerance) or
                (prev_y > self.stop_y + self.tolerance and curr_y < self.stop_y - self.tolerance)):
                
                # Calculate approximate crossing point (linear interpolation)
                if prev_y != curr_y:  # Avoid division by zero
                    t = (self.stop_y - prev_y) / (curr_y - prev_y)
                    cross_x = prev_x + t * (curr_x - prev_x)
                    return True, (cross_x, self.stop_y)
                else:
                    return True, (prev_x, self.stop_y)
            
            return False, None
        
        return False, None
        
    def _line_intersection(self, line1, line2):
        """
        Determine if two line segments intersect
        line1 and line2 are in format (x1, y1, x2, y2)
        Returns True if the lines intersect, False otherwise
        """
        # Convert line segments to parametric form
        def line_to_params(line):
            # Make sure we only have 4 values for the line
            if len(line) > 4:
                print(f"Warning: Line has {len(line)} values, expected 4. Using first 4 values.")
                x1, y1, x2, y2 = line[:4]
            else:
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