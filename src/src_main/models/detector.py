# models/detector.py

import cv2
import numpy as np
import torch
from ultralytics import YOLO

class VehicleDetector:
    """
    Vehicle detection class to encapsulate detection operations
    """
    def __init__(self, model_path, conf_threshold=0.65, iou_threshold=0.5, 
                 device=None, use_tensorrt=False):
        """
        Initialize the vehicle detector
        
        Args:
            model_path: Path to the YOLO model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cuda' or 'cpu')
            use_tensorrt: Whether to convert the model to TensorRT
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Set device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Load model
        self.model = self._load_model(use_tensorrt)
        
    def _load_model(self, use_tensorrt=False):
        """Load the detection model with optional TensorRT conversion"""
        model = YOLO(self.model_path, task="detect")
        
        if use_tensorrt and torch.cuda.is_available():
            # TensorRT conversion would go here
            # This is simplified - full implementation would check for engine files
            print("TensorRT conversion not implemented in this example")
            
        return model
        
    def detect(self, frame):
        """Detect vehicles in a frame"""
        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device
            )[0]
            return results
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("Warning: CUDA out of memory for detection, falling back to CPU")
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                    device="cpu"
                )[0]
                return results
            else:
                raise
                
    def detect_and_track(self, frame, tracker_type="bytetrack", persist=True):
        """Detect and track vehicles in a frame"""
        try:
            # Always use custom tracker config with necessary parameters
            custom_tracker_path = "/home/quangsang/Study/maiAnhEm/soict-hackathon-2024/src/src_main/custom_bytetrack.yaml"
            
            results = self.model.track(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device,
                tracker=custom_tracker_path,
                persist=persist
            )[0]
            return results
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("Warning: CUDA out of memory for tracking, falling back to CPU")
                results = self.model.track(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                    device="cpu",
                    tracker=custom_tracker_path,
                    persist=persist
                )[0]
                return results
            else:
                raise