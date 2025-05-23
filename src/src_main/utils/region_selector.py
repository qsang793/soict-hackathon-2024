#!/usr/bin/env python
# region_selector.py - A utility to select regions for traffic violation detection

import argparse
import cv2
import numpy as np
import os

class RegionSelector:
    def __init__(self, video_path, output_path=None):
        """Initialize the region selector tool"""
        self.video_path = video_path
        self.output_path = output_path
        self.frame = None
        self.original_frame = None
        self.drawing = False
        self.roi_points = []
        self.current_roi = []
        self.regions = {
            'stop_line': [],
            'green_zone': [],
            'red_zone': [],
            'traffic_light_bbox': []
        }
        self.current_region_type = 'stop_line'
        self.roi_color_map = {
            'stop_line': (0, 255, 255),    # Yellow
            'green_zone': (0, 255, 0),     # Green
            'red_zone': (0, 0, 255),       # Red
            'traffic_light_bbox': (255, 0, 255)  # Magenta
        }
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for selecting regions"""
        # Make a copy of the frame to draw on
        temp_frame = self.frame.copy()
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing
            self.drawing = True
            self.current_roi.append((x, y))
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Draw line from last point to current mouse position
            if len(self.current_roi) > 0:
                cv2.line(temp_frame, self.current_roi[-1], (x, y), 
                        self.roi_color_map[self.current_region_type], 2)
                
                # If this is a rectangle (traffic_light_bbox or stop_line), draw preview
                if self.current_region_type in ['traffic_light_bbox', 'stop_line'] and len(self.current_roi) == 1:
                    start_point = self.current_roi[0]
                    cv2.rectangle(temp_frame, start_point, (x, y), 
                                 self.roi_color_map[self.current_region_type], 2)
                
                # Display the temporary drawing
                cv2.imshow('Region Selector', temp_frame)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                # Handle rectangle types (traffic_light_bbox, stop_line with 2 points)
                if self.current_region_type == 'traffic_light_bbox':
                    if len(self.current_roi) == 1:
                        # Complete the rectangle
                        x1, y1 = self.current_roi[0]
                        self.regions[self.current_region_type] = [x1, y1, x, y]
                        self.current_roi = []
                        self.drawing = False
                        self.redraw_frame()
                        print(f"{self.current_region_type} coordinates: {self.regions[self.current_region_type]}")
                        
                elif self.current_region_type == 'stop_line':
                    if len(self.current_roi) == 1:
                        # For stop line, we just need 2 points
                        self.current_roi.append((x, y))
                        self.regions[self.current_region_type] = [
                            self.current_roi[0][0], self.current_roi[0][1],
                            self.current_roi[1][0], self.current_roi[1][1]
                        ]
                        self.current_roi = []
                        self.drawing = False
                        self.redraw_frame()
                        print(f"{self.current_region_type} coordinates: {self.regions[self.current_region_type]}")
                
                # For polygon regions (green_zone, red_zone)
                elif self.current_region_type in ['green_zone', 'red_zone']:
                    self.current_roi.append((x, y))
                    # Draw the current point
                    cv2.circle(self.frame, (x, y), 3, self.roi_color_map[self.current_region_type], -1)
                    
                    # Connect to previous point if exists
                    if len(self.current_roi) > 1:
                        cv2.line(self.frame, self.current_roi[-2], self.current_roi[-1], 
                                self.roi_color_map[self.current_region_type], 2)
                    
                    # Show the updated frame
                    cv2.imshow('Region Selector', self.frame)
                    
                    self.drawing = False
    
    def redraw_frame(self):
        """Redraw the frame with all selected regions"""
        self.frame = self.original_frame.copy()
        
        # Draw stop line
        if len(self.regions['stop_line']) == 4:
            x1, y1, x2, y2 = self.regions['stop_line']
            cv2.line(self.frame, (x1, y1), (x2, y2), self.roi_color_map['stop_line'], 2)
            
        # Draw traffic light bbox
        if len(self.regions['traffic_light_bbox']) == 4:
            x1, y1, x2, y2 = self.regions['traffic_light_bbox']
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), self.roi_color_map['traffic_light_bbox'], 2)
            
        # Draw zones (polygons)
        for zone_type in ['green_zone', 'red_zone']:
            points = self.regions[zone_type]
            if len(points) >= 6:  # At least 3 points (x1,y1,x2,y2,x3,y3)
                # Convert to numpy array format for polygon drawing
                pts = np.array([(points[i], points[i+1]) for i in range(0, len(points), 2)], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(self.frame, [pts], True, self.roi_color_map[zone_type], 2)
                
        # Draw current incomplete polygon
        if self.current_region_type in ['green_zone', 'red_zone'] and len(self.current_roi) > 0:
            for i in range(len(self.current_roi)):
                cv2.circle(self.frame, self.current_roi[i], 3, self.roi_color_map[self.current_region_type], -1)
                if i > 0:
                    cv2.line(self.frame, self.current_roi[i-1], self.current_roi[i], 
                            self.roi_color_map[self.current_region_type], 2)
        
        cv2.imshow('Region Selector', self.frame)
    
    def format_coordinates(self, region_type):
        """Format coordinates for command line arguments"""
        if region_type == 'traffic_light_bbox' or region_type == 'stop_line':
            if len(self.regions[region_type]) == 4:
                return ','.join(map(str, self.regions[region_type]))
            return None
            
        elif region_type in ['green_zone', 'red_zone']:
            if len(self.regions[region_type]) >= 6:
                return ','.join(map(str, self.regions[region_type]))
            return None
    
    def run(self):
        """Run the region selector tool"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return False
        
        # Read the first frame
        ret, self.frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            return False
            
        self.original_frame = self.frame.copy()
        
        # Set up the window and mouse callback
        cv2.namedWindow('Region Selector')
        cv2.setMouseCallback('Region Selector', self.mouse_callback)
        
        print("\nRegion Selector Tool")
        print("====================")
        print("Instructions:")
        print("  - Press 1: Select stop line (click two points to define a line)")
        print("  - Press 2: Select green zone (click multiple points, then press 'c' to close polygon)")
        print("  - Press 3: Select red zone (click multiple points, then press 'c' to close polygon)")
        print("  - Press 4: Select traffic light bounding box (click and drag)")
        print("  - Press 'c': Complete current polygon")
        print("  - Press 'r': Reset current region")
        print("  - Press 's': Save all regions and show command")
        print("  - Press 'q': Quit without saving\n")
        
        while True:
            cv2.imshow('Region Selector', self.frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
                
            elif key == ord('1'):
                self.current_region_type = 'stop_line'
                self.current_roi = []
                print("Selected: Stop Line")
                
            elif key == ord('2'):
                self.current_region_type = 'green_zone'
                self.current_roi = []
                print("Selected: Green Zone")
                
            elif key == ord('3'):
                self.current_region_type = 'red_zone'
                self.current_roi = []
                print("Selected: Red Zone")
                
            elif key == ord('4'):
                self.current_region_type = 'traffic_light_bbox'
                self.current_roi = []
                print("Selected: Traffic Light Bounding Box")
                1
            elif key == ord('r'):
                # Reset the current region
                if self.current_region_type:
                    self.regions[self.current_region_type] = []
                    self.current_roi = []
                    self.redraw_frame()
                    print(f"Reset {self.current_region_type}")
                
            elif key == ord('c'):
                # Complete the current polygon
                if self.current_region_type in ['green_zone', 'red_zone'] and len(self.current_roi) >= 3:
                    # Extract x,y coordinates as flat list [x1,y1,x2,y2,...]
                    flat_coords = []
                    for point in self.current_roi:
                        flat_coords.extend(point)
                    
                    self.regions[self.current_region_type] = flat_coords
                    self.current_roi = []
                    self.redraw_frame()
                    print(f"{self.current_region_type} coordinates: {self.regions[self.current_region_type]}")
                
            elif key == ord('s'):
                # Generate command and save frame if desired
                if self.output_path:
                    cv2.imwrite(self.output_path, self.frame)
                    print(f"Saved annotated frame to {self.output_path}")
                
                # Generate command string
                cmd = "python traffic_violation_detector.py"
                
                for region_type in ['stop_line', 'green_zone', 'red_zone', 'traffic_light_bbox']:
                    coords = self.format_coordinates(region_type)
                    if coords:
                        cmd += f" --{region_type} {coords}"
                
                # Add default arguments
                cmd += " --input <VIDEO_PATH> --output_video result_violation.mp4 --output_data violations_data.csv"
                cmd += " --detection_method combined --min_frames_in_green 5 --trajectory_interpolation 15"
                cmd += " --tolerance 20 --save_preview_frames 10"
                
                print("\nGenerated Command:")
                print(cmd)
                print("\nCopy and modify the command above to run your detection.")
                
                if self.output_path:
                    # Save the command to a text file
                    cmd_file = os.path.splitext(self.output_path)[0] + "_command.txt"
                    with open(cmd_file, 'w') as f:
                        f.write(cmd)
                    print(f"Command also saved to {cmd_file}")
        
        cap.release()
        cv2.destroyAllWindows()
        return True


def main():
    parser = argparse.ArgumentParser(description="Select regions for traffic violation detection")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to the input video")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the annotated frame (optional)")
    
    args = parser.parse_args()
    
    selector = RegionSelector(args.input, args.output)
    selector.run()


if __name__ == "__main__":
    main()
