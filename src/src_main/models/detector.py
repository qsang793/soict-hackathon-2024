# models/detector.py

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class VehicleDetector:
    """
    Vehicle detection class to encapsulate detection operations
    """
    def __init__(self, model_path, conf_threshold=0.65, iou_threshold=0.5, 
                 device=None, args=None):
        """
        Initialize the vehicle detector
        
        Args:
            model_path: Path to the YOLO model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cuda' or 'cpu')
            args: Command line arguments containing TensorRT settings
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.args = args
        
        # Set device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Load model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the detection model with optional TensorRT conversion"""
        use_tensorrt = getattr(self.args, 'use_tensorrt', False)

        if use_tensorrt and torch.cuda.is_available():
            print("🚀 Attempting to use TensorRT for vehicle detection model...")
            task = "detect"
            
            # Construct engine path
            basename = os.path.splitext(self.model_path)[0]
            engine_path = f"{basename}_{task}.engine"
            
            # If engine file exists, load it
            if os.path.exists(engine_path):
                print(f"✅ Found existing TensorRT engine: {engine_path}. Verifying compatibility...")
                try:
                    model = YOLO(engine_path, task=task)
                    # Verify the engine by running a dummy prediction
                    _ = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
                    print("✅ Engine is compatible and loaded successfully.")
                    return model
                except Exception as e:
                    print(f"⚠️ Incompatible or corrupt TensorRT engine detected: {e}.")
                    print(f"🗑️ Deleting invalid engine file: {engine_path}")
                    try:
                        os.remove(engine_path)
                    except OSError as remove_error:
                        print(f"🔥 Error deleting engine file: {remove_error}. Please delete it manually and restart.")
                        # If we can't delete, we must fall back to PyTorch
                        print("Falling back to PyTorch model.")
                        return YOLO(self.model_path, task="detect")

            # If engine file does not exist (or was just deleted), create it
            print(f"🛠️ No TensorRT engine found. Converting {self.model_path} to TensorRT...")
            model = YOLO(self.model_path, task=task)
            
            try:
                model.export(
                    format='engine', 
                    half=getattr(self.args, 'half_precision', False), 
                    workspace=getattr(self.args, 'tensorrt_workspace', 8), 
                    device=self.device,
                    dynamic=getattr(self.args, 'tensorrt_dynamic', False)
                )
                
                # The exported file name is based on the original model name
                default_export_path = self.model_path.replace('.pt', '.engine')
                
                # Rename to our standard format if needed
                if os.path.exists(default_export_path) and default_export_path != engine_path:
                    os.rename(default_export_path, engine_path)
                    
                if os.path.exists(engine_path):
                    print(f"✅ Conversion successful. Engine saved at: {engine_path}")
                    return YOLO(engine_path, task=task)
                else:
                    print(f"⚠️ TensorRT engine not found after conversion. Falling back to PyTorch model.")
            
            except Exception as e:
                print(f"❌ Error converting model to TensorRT: {e}")
                print("Falling back to PyTorch model.")
                return model # Return original PyTorch model on failure

        # Default: load PyTorch model
        print("Using standard PyTorch model for vehicle detection.")
        model = YOLO(self.model_path, task="detect")
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