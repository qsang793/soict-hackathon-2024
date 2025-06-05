# models/tracker.py

import numpy as np

class VehicleTracker:
    """
    Handles vehicle tracking data and trajectory management
    """
    def __init__(self, max_trajectory_length=50, speed_estimation_params=None):
        """
        Initialize the vehicle tracker
        
        Args:
            max_trajectory_length: Maximum number of points to keep in trajectory history
            speed_estimation_params: Parameters for speed estimation (fps, pixels_per_meter)
        """
        self.trajectories = {}  # Track data keyed by track_id
        self.max_trajectory_length = max_trajectory_length
        
        # Speed estimation parameters
        if speed_estimation_params is None:
            self.speed_estimation_params = {
                'fps': 30,
                'pixels_per_meter': 10,
                'max_speed': 150  # Maximum speed in km/h
            }
        else:
            self.speed_estimation_params = speed_estimation_params
            
    def update(self, detection_results, frame_count):
        """
        Update trajectories with new detections
        
        Args:
            detection_results: YOLO detection results with tracking
            frame_count: Current frame count
            
        Returns:
            List of current detections with trajectory info
        """
        # Process tracking results
        if not hasattr(detection_results, 'boxes') or len(detection_results.boxes) == 0:
            return []
            
        # Check if tracking is enabled
        if not hasattr(detection_results.boxes, 'id') or detection_results.boxes.id is None:
            print("Warning: Tracking not available for this frame")
            return []
            
        # Get detection data
        boxes = detection_results.boxes.xyxy.cpu().numpy()
        track_ids = detection_results.boxes.id.int().cpu().numpy()
        labels = detection_results.boxes.cls.int().cpu().numpy()
        scores = detection_results.boxes.conf.cpu().numpy()
        
        # Create detection records for this frame
        detections = []
        
        for i, (box, track_id, label, score) in enumerate(zip(boxes, track_ids, labels, scores)):
            # Calculate center point for trajectory
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            
            # Create or update trajectory data
            if track_id not in self.trajectories:
                self.trajectories[track_id] = {
                    "history": [(center_x, center_y)],
                    "label": label,
                    "first_seen": frame_count,
                    "last_seen": frame_count,
                    "track_id": int(track_id)
                }
            else:
                # Update trajectory
                self.trajectories[track_id]["history"].append((center_x, center_y))
                self.trajectories[track_id]["last_seen"] = frame_count
                
                # Limit history length
                if len(self.trajectories[track_id]["history"]) > self.max_trajectory_length:
                    self.trajectories[track_id]["history"] = self.trajectories[track_id]["history"][-self.max_trajectory_length:]
            
            # Calculate speed
            speed = self._calculate_speed(track_id)
            
            # Add to detections list
            detections.append({
                "box": box,
                "track_id": int(track_id),
                "label": int(label),
                "score": float(score),
                "center": (center_x, center_y),
                "speed": speed
            })
            
        return detections
        
    def _calculate_speed(self, track_id, window_size=2):
        """
        Calculate vehicle speed based on recent trajectory points
        
        Args:
            track_id: Vehicle track ID
            window_size: Number of recent points to use for calculation
            
        Returns:
            Speed in km/h
        """
        if track_id not in self.trajectories or "history" not in self.trajectories[track_id]:
            return 0
            
        history = self.trajectories[track_id]["history"]
        if len(history) < 2:
            return 0
            
        # Use the most recent points for calculation
        recent_points = history[-window_size:] if len(history) >= window_size else history
        
        # Calculate average distance between consecutive points
        distances = []
        for i in range(1, len(recent_points)):
            p1 = recent_points[i-1]
            p2 = recent_points[i]
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            distances.append(distance)
            
        if not distances:
            return 0
            
        avg_distance_pixels = sum(distances) / len(distances)
        
        # Convert to meters
        fps = self.speed_estimation_params['fps']
        pixels_per_meter = self.speed_estimation_params['pixels_per_meter']
        max_speed = self.speed_estimation_params['max_speed']
        
        distance_meters = avg_distance_pixels / pixels_per_meter
        
        # Calculate time between frames in hours
        time_hours = 1 / (fps * 3600)
        
        # Calculate speed in km/h
        speed_kmh = (distance_meters / 1000) / time_hours
        
        # Cap at maximum speed to filter outliers
        return min(speed_kmh, max_speed)
        
    def get_trajectory(self, track_id, max_points=None):
        """Get trajectory for a specific track ID"""
        if track_id not in self.trajectories:
            return []
            
        history = self.trajectories[track_id]["history"]
        if max_points is not None and len(history) > max_points:
            return history[-max_points:]
        return history
        
    def get_active_tracks(self, current_frame, max_age=30):
        """Get IDs of currently active tracks"""
        return [tid for tid, track in self.trajectories.items() 
               if track["last_seen"] >= current_frame - max_age]
               
    def interpolate_trajectory(self, track_id, steps=5):
        """
        Create interpolated points between the last two trajectory points
        Useful for more accurate violation detection with fast-moving vehicles
        
        Args:
            track_id: Vehicle track ID
            steps: Number of interpolation steps
            
        Returns:
            List of interpolated points including endpoints
        """
        if track_id not in self.trajectories:
            return []
            
        history = self.trajectories[track_id]["history"]
        if len(history) < 2:
            return history
            
        # Get the last two points
        p1 = history[-2]
        p2 = history[-1]
        
        # Create interpolated points
        points = []
        for i in range(steps):
            t = i / (steps - 1)
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            points.append((x, y))
            
        return points