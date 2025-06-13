#!/usr/bin/env python3
"""
Simple VLM Addon for Traffic Violation Detector
- Runs in background thread
- Does not affect main detection performance
- Simple batch processing
"""

import os
import sys
import time
import threading
import json
from queue import Queue
from typing import Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image

# Add project root to path for VLM imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src_vlm.gemini import GeminiClient
    from src_vlm.io_utils import read_text_file
    from src_vlm.main import VehicleInformation
    VLM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  VLM not available: {e}")
    VLM_AVAILABLE = False


class SimpleVLMProcessor:
    """
    Simple background VLM processor that doesn't affect main detection performance
    """
    
    def __init__(self, api_key: str = None, batch_size: int = 5, enabled: bool = True, output_dir: str = "violation_images"):
        """
        Initialize VLM processor
        
        Args:
            api_key: Gemini API key
            batch_size: Number of images to process in one batch
            enabled: Whether to enable VLM processing
            output_dir: Directory to save violation images
        """
        self.enabled = enabled and VLM_AVAILABLE
        
        if not self.enabled:
            print("🔄 VLM processing disabled")
            return
            
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.batch_size = batch_size
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize VLM client
        try:
            self.vlm_client = GeminiClient(
                model_name="gemini-2.0-flash-lite",
                api_key=self.api_key
            )
            
            # Load prompt
            prompt_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src_vlm', 'prompt.md')
            self.system_prompt = read_text_file(prompt_path)
            
            print("✅ VLM processor initialized")
        except Exception as e:
            print(f"❌ VLM initialization failed: {e}")
            self.enabled = False
            return
        
        # Background processing
        self.queue = Queue()
        self.results = {}
        self.processing_thread = None
        self.running = False
        
        # Start background thread
        self._start_background_processing()
    
    def _start_background_processing(self):
        """Start background processing thread"""
        if not self.enabled:
            return
            
        self.running = True
        self.processing_thread = threading.Thread(target=self._background_processor, daemon=True)
        self.processing_thread.start()
        print("🔄 VLM background processing started")
    
    def add_violation(self, frame: np.ndarray, bbox: list, track_id: int, frame_count: int, violation_info: dict):
        """
        Add a violation for VLM processing (non-blocking)
        
        Args:
            frame: Video frame
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: Vehicle track ID
            frame_count: Current frame count
            violation_info: Violation information
        """
        if not self.enabled:
            return
            
        try:
            # Crop vehicle image
            x1, y1, x2, y2 = bbox
            
            # Convert to integers (fix for float/numpy array bbox)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Validate bbox
            h, w = frame.shape[:2]
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                print(f"⚠️  Invalid bbox for violation {track_id}: {bbox}")
                return
            
            # Add padding
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Crop image
            cropped = frame[y1:y2, x1:x2].copy()
            
            # Check if cropped image is valid
            if cropped.size == 0:
                print(f"⚠️  Empty cropped image for violation {track_id}")
                return
            
            # Save cropped image with maximum quality
            image_path = os.path.join(self.output_dir, f"violation_{track_id}_{frame_count}.jpg")
            cv2.imwrite(image_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 100])  # Chất lượng tối đa 100%
            
            # Add to processing queue (non-blocking)
            violation_data = {
                'track_id': track_id,
                'frame_count': frame_count,
                'image_path': image_path,
                'cropped_image': cropped,
                'violation_info': violation_info,
                'timestamp': time.time()
            }
            
            try:
                self.queue.put(violation_data, block=False)  # Non-blocking
                print(f"📸 Added violation {track_id} to VLM queue")
            except:
                print(f"⚠️  VLM queue full, skipping violation {track_id}")
                
        except Exception as e:
            print(f"❌ Error processing violation {track_id}: {e}")
            # Don't crash the main detection process
    
    def _background_processor(self):
        """Background thread for processing violations"""
        while self.running or not self.queue.empty():
            current_batch = []
            try:
                # Build a batch up to batch_size
                while len(current_batch) < self.batch_size:
                    item = self.queue.get(timeout=1) # Wait for 1 second
                    current_batch.append(item)
            except Exception: # Queue empty or timeout
                pass

            if current_batch:
                self._process_batch(current_batch)
            
            # If not running and queue is empty, we can exit the thread
            if not self.running and self.queue.empty():
                break
    
    def _process_batch(self, batch):
        """Process a batch of violations"""
        if not batch:
            return
            
        print(f"🔄 Processing VLM batch of {len(batch)} violations...")
        
        try:
            # Prepare images
            images = []
            for item in batch:
                pil_image = Image.fromarray(cv2.cvtColor(item['cropped_image'], cv2.COLOR_BGR2RGB))
                images.append(pil_image)
            
            # Send to VLM
            messages = [self.system_prompt] + images
            response, usage = self.vlm_client.request(
                messages=messages,
                response_mime_type="application/json",
                response_schema=list[VehicleInformation],
            )
            
            if response:
                self._store_results(batch, response)
                print(f"✅ VLM batch processed successfully")
            else:
                print(f"❌ VLM batch processing failed")
                
        except Exception as e:
            print(f"❌ VLM batch error: {e}")
    
    def _store_results(self, batch, response):
        """Store VLM results"""
        try:
            results = json.loads(response)
            
            for item, result in zip(batch, results):
                track_id = item['track_id']
                frame_count = item['frame_count']
                image_filename = os.path.basename(item['image_path'])
                
                self.results[track_id] = {
                    'license_plate': result.get('license_plate'),
                    'description': result.get('description'),
                    'violation_info': item['violation_info'],
                    'image_path': item['image_path'],
                    'processed_at': time.time()
                }
                
                plate = result.get('license_plate', 'Not detected')
                vlm_data = {
                    'track_id': track_id,
                    'frame_count': frame_count,
                    'license_plate': plate,
                    'vehicle_description': result.get('description'),
                    'image_filename': image_filename
                }
                print(f"VLM_RESULT: {json.dumps(vlm_data)}")
                sys.stdout.flush()

        except Exception as e:
            print(f"❌ Error storing VLM results: {e}")
    
    def get_result(self, track_id: int) -> Optional[Dict]:
        """Get VLM result for a track ID"""
        return self.results.get(track_id)
    
    def get_all_results(self) -> Dict[int, Dict]:
        """Get all VLM results"""
        return self.results.copy()
    
    def export_report(self, output_path: str = "violations_with_plates.json"):
        """Export comprehensive report"""
        if not self.enabled:
            return
            
        report = {
            'total_violations': len(self.results),
            'violations': []
        }
        
        for track_id, data in self.results.items():
            violation = {
                'track_id': track_id,
                'license_plate': data['license_plate'],
                'vehicle_description': data['description'],
                'violation_type': data['violation_info'].get('violation_type'),
                'frame_count': data['violation_info'].get('frame_count'),
                'image_path': data['image_path']
            }
            report['violations'].append(violation)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 VLM report exported: {output_path}")
        print(f"📊 Total violations: {report['total_violations']}")
        plates_found = sum(1 for v in report['violations'] if v['license_plate'])
        print(f"🔍 License plates found: {plates_found}")
    
    def stop(self):
        """Stop background processing and wait for queue to empty"""
        if self.enabled and self.running:
            print("🛑 VLM shutting down, waiting for all violations to be processed...")
            self.running = False
            if self.processing_thread:
                self.processing_thread.join(timeout=60) # Wait up to 60s for remaining API calls
            print("🛑 VLM processing stopped.")


# Simple integration function
def add_vlm_to_traffic_system():
    """
    Simple function to add VLM processing to existing traffic violation detector
    """
    print("🔧 Adding VLM support to traffic violation detector...")
    
    # Initialize VLM processor
    vlm_processor = SimpleVLMProcessor(
        api_key="AIzaSyDSF1qp4vK5AkAIKwO1_3kkaolkLxRomVU",
        batch_size=5,
        enabled=True
    )
    
    return vlm_processor


if __name__ == "__main__":
    # Test the VLM addon
    print("🧪 Testing VLM Addon...")
    
    processor = SimpleVLMProcessor(
        api_key="AIzaSyDSF1qp4vK5AkAIKwO1_3kkaolkLxRomVU",
        batch_size=3,
        enabled=True
    )
    
    print("✅ VLM Addon test completed")
    processor.stop() 