# violations/detector.py

from abc import ABC, abstractmethod

class ViolationDetector(ABC):
    """
    Abstract base class for violation detectors
    """
    @abstractmethod
    def check_violation(self, tracker, track_id, frame_count, is_red_phase):
        """
        Check if a vehicle has committed a violation
        
        Args:
            tracker: The vehicle tracker instance
            track_id: The track ID to check
            frame_count: Current frame count
            is_red_phase: Whether the traffic light is currently red
            
        Returns:
            (violated, violation_info): Tuple with violation status and info
        """
        pass
        
        






