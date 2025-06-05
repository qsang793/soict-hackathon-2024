# violations/factory.py

from .stop_line import StopLineViolationDetector
from .zone import ZoneViolationDetector

class ViolationDetectorFactory:
    """
    Factory class for creating violation detectors
    """
    @staticmethod
    def create(method, config):
        """
        Create a violation detector based on method and config
        
        Args:
            method: 'stop_line', 'zone', or 'combined'
            config: Configuration dictionary
            
        Returns:
            List of violation detector instances
        """
        detectors = []
        
        if method in ['stop_line', 'combined']:
            stop_line_detector = StopLineViolationDetector(
                stop_line=config.get('stop_line'),
                stop_y=config.get('stop_y'),
                tolerance=config.get('tolerance', 10),
                interpolation_steps=config.get('interpolation_steps', 5)
            )
            detectors.append(stop_line_detector)
            
        if method in ['zone', 'combined']:
            zone_detector = ZoneViolationDetector(
                green_zone=config.get('green_zone'),
                red_zone=config.get('red_zone'),
                min_frames_in_green=config.get('min_frames_in_green', 3)
            )
            detectors.append(zone_detector)
            
        return detectors