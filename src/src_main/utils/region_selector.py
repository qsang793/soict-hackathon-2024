#!/usr/bin/env python
# region_selector.py - A utility to select regions for traffic violation detection with video resize support

import argparse
import cv2
import numpy as np
import os

class RegionSelector:
    def __init__(self, video_path, output_path=None, target_width=None, target_height=None, scale_factor=None, use_resized_coords=False):
        """Initialize the region selector tool"""
        self.video_path = video_path
        self.output_path = output_path
        self.frame = None
        self.original_frame = None
        self.display_frame = None  # Frame used for display (potentially resized)
        self.drawing = False
        self.roi_points = []
        self.current_roi = []
        
        # Video dimensions
        self.original_width = None
        self.original_height = None
        self.display_width = None
        self.display_height = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        
        # Resize parameters
        self.target_width = target_width
        self.target_height = target_height
        self.scale_factor = scale_factor
        self.use_resized_coords = use_resized_coords  # NEW: Option to save resized coordinates
        
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
    
    def calculate_resize_parameters(self, original_width, original_height):
        """Calculate resize parameters and scaling factors"""
        self.original_width = original_width
        self.original_height = original_height
        
        # Determine target dimensions
        if self.scale_factor:
            # Use scale factor
            self.display_width = int(original_width * self.scale_factor)
            self.display_height = int(original_height * self.scale_factor)
        elif self.target_width and self.target_height:
            # Use specific dimensions
            self.display_width = self.target_width
            self.display_height = self.target_height
        elif self.target_width:
            # Scale based on width, maintain aspect ratio
            self.display_width = self.target_width
            aspect_ratio = original_height / original_width
            self.display_height = int(self.target_width * aspect_ratio)
        elif self.target_height:
            # Scale based on height, maintain aspect ratio
            self.display_height = self.target_height
            aspect_ratio = original_width / original_height
            self.display_width = int(self.target_height * aspect_ratio)
        else:
            # No resize, use original dimensions
            self.display_width = original_width
            self.display_height = original_height
        
        # Calculate scaling factors
        self.scale_x = self.display_width / original_width
        self.scale_y = self.display_height / original_height
        
        print(f"Original dimensions: {original_width}x{original_height}")
        print(f"Display dimensions: {self.display_width}x{self.display_height}")
        print(f"Scale factors: x={self.scale_x:.3f}, y={self.scale_y:.3f}")
        
        # NEW: Print coordinate system being used
        coords_type = "Resized" if self.use_resized_coords else "Original"
        coords_dims = f"{self.display_width}x{self.display_height}" if self.use_resized_coords else f"{original_width}x{original_height}"
        print(f"Coordinate system: {coords_type} ({coords_dims})")
    
    def display_to_original_coords(self, x, y):
        """Convert display coordinates to original video coordinates"""
        orig_x = int(x / self.scale_x)
        orig_y = int(y / self.scale_y)
        return orig_x, orig_y
    
    def original_to_display_coords(self, x, y):
        """Convert original coordinates to display coordinates"""
        disp_x = int(x * self.scale_x)
        disp_y = int(y * self.scale_y)
        return disp_x, disp_y
    
    def resize_frame_for_display(self, frame):
        """Resize frame for display purposes"""
        if self.scale_x != 1.0 or self.scale_y != 1.0:
            return cv2.resize(frame, (self.display_width, self.display_height))
        return frame.copy()
    
    def get_coords_to_save(self, display_x, display_y):
        """Get coordinates in the format specified by use_resized_coords"""
        if self.use_resized_coords:
            # Save display coordinates (resized)
            return display_x, display_y
        else:
            # Save original coordinates (as before)
            return self.display_to_original_coords(display_x, display_y)
    
    def coords_for_display(self, saved_x, saved_y):
        """Convert saved coordinates back to display coordinates"""
        if self.use_resized_coords:
            # Saved coords are already display coords
            return saved_x, saved_y
        else:
            # Saved coords are original coords, need to convert
            return self.original_to_display_coords(saved_x, saved_y)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for selecting regions"""
        
        # Get coordinates to save based on coordinate system preference
        save_x, save_y = self.get_coords_to_save(x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # For rectangle types (traffic_light_bbox, stop_line) - start drag operation
            if self.current_region_type in ['traffic_light_bbox', 'stop_line']:
                self.drawing = True
                self.current_roi = [(save_x, save_y)]
                
            # For polygon types (green_zone, red_zone) - add point to polygon
            elif self.current_region_type in ['green_zone', 'red_zone']:
                self.current_roi.append((save_x, save_y))
                
                # Draw the current point
                cv2.circle(self.display_frame, (x, y), 3, self.roi_color_map[self.current_region_type], -1)
                
                # Connect to previous point if exists
                if len(self.current_roi) > 1:
                    prev_save_x, prev_save_y = self.current_roi[-2]
                    prev_disp_x, prev_disp_y = self.coords_for_display(prev_save_x, prev_save_y)
                    cv2.line(self.display_frame, (prev_disp_x, prev_disp_y), (x, y), 
                            self.roi_color_map[self.current_region_type], 2)
                
                # Show the updated frame
                cv2.imshow('Region Selector', self.display_frame)
                coords_type = "Resized" if self.use_resized_coords else "Original"
                print(f"Added point {len(self.current_roi)}: ({save_x}, {save_y}) [{coords_type} coords]")
                
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Only for rectangle types during drag operation
            if self.current_region_type in ['traffic_light_bbox', 'stop_line']:
                # Make a copy of the display frame to draw preview
                temp_frame = self.display_frame.copy()
                
                if len(self.current_roi) > 0:
                    start_save_x, start_save_y = self.current_roi[0]
                    start_disp_x, start_disp_y = self.coords_for_display(start_save_x, start_save_y)
                    cv2.rectangle(temp_frame, (start_disp_x, start_disp_y), (x, y), 
                                self.roi_color_map[self.current_region_type], 2)
                    
                    # Display the temporary drawing
                    cv2.imshow('Region Selector', temp_frame)
                    
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            # Only for rectangle types - complete the rectangle
            if self.current_region_type in ['traffic_light_bbox', 'stop_line']:
                if len(self.current_roi) == 1:
                    # Complete the rectangle
                    x1, y1 = self.current_roi[0]
                    x2, y2 = save_x, save_y
                    # Ensure we have valid rectangle coordinates
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    self.regions[self.current_region_type] = [x1, y1, x2, y2]
                    self.current_roi = []
                    self.drawing = False
                    self.redraw_frame()
                    coords_type = "Resized" if self.use_resized_coords else "Original"
                    print(f"{self.current_region_type} coordinates: {self.regions[self.current_region_type]} [{coords_type} coords]")
                    
        elif event == cv2.EVENT_MOUSEMOVE:
            # Show preview line for polygon types (when not clicking)
            if (self.current_region_type in ['green_zone', 'red_zone'] and 
                len(self.current_roi) > 0 and not self.drawing):
                # Make a copy of the display frame to draw preview
                temp_frame = self.display_frame.copy()
                
                # Redraw existing points and lines
                for i in range(len(self.current_roi)):
                    save_pt_x, save_pt_y = self.current_roi[i]
                    disp_pt_x, disp_pt_y = self.coords_for_display(save_pt_x, save_pt_y)
                    cv2.circle(temp_frame, (disp_pt_x, disp_pt_y), 3, 
                             self.roi_color_map[self.current_region_type], -1)
                    if i > 0:
                        prev_save_x, prev_save_y = self.current_roi[i-1]
                        prev_disp_x, prev_disp_y = self.coords_for_display(prev_save_x, prev_save_y)
                        cv2.line(temp_frame, (prev_disp_x, prev_disp_y), (disp_pt_x, disp_pt_y), 
                               self.roi_color_map[self.current_region_type], 2)
                
                # Draw preview line from last point to cursor
                if len(self.current_roi) > 0:
                    last_save_x, last_save_y = self.current_roi[-1]
                    last_disp_x, last_disp_y = self.coords_for_display(last_save_x, last_save_y)
                    cv2.line(temp_frame, (last_disp_x, last_disp_y), (x, y), 
                           self.roi_color_map[self.current_region_type], 1)
                
                cv2.imshow('Region Selector', temp_frame)
    
    def redraw_frame(self):
        """Redraw the frame with all selected regions"""
        self.display_frame = self.resize_frame_for_display(self.original_frame)
        
        # Draw stop line
        if len(self.regions['stop_line']) == 4:
            x1, y1, x2, y2 = self.regions['stop_line']
            # Convert to display coordinates
            disp_x1, disp_y1 = self.coords_for_display(x1, y1)
            disp_x2, disp_y2 = self.coords_for_display(x2, y2)
            cv2.line(self.display_frame, (disp_x1, disp_y1), (disp_x2, disp_y2), 
                    self.roi_color_map['stop_line'], 2)
            
        # Draw traffic light bbox
        if len(self.regions['traffic_light_bbox']) == 4:
            x1, y1, x2, y2 = self.regions['traffic_light_bbox']
            # Convert to display coordinates
            disp_x1, disp_y1 = self.coords_for_display(x1, y1)
            disp_x2, disp_y2 = self.coords_for_display(x2, y2)
            cv2.rectangle(self.display_frame, (disp_x1, disp_y1), (disp_x2, disp_y2), 
                         self.roi_color_map['traffic_light_bbox'], 2)
            
        # Draw zones (polygons)
        for zone_type in ['green_zone', 'red_zone']:
            points = self.regions[zone_type]
            if len(points) >= 6:  # At least 3 points (x1,y1,x2,y2,x3,y3)
                # Convert to display coordinates and numpy array format
                display_points = []
                for i in range(0, len(points), 2):
                    save_x, save_y = points[i], points[i+1]
                    disp_x, disp_y = self.coords_for_display(save_x, save_y)
                    display_points.append((disp_x, disp_y))
                
                pts = np.array(display_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(self.display_frame, [pts], True, self.roi_color_map[zone_type], 2)
                
        # Draw current incomplete polygon
        if self.current_region_type in ['green_zone', 'red_zone'] and len(self.current_roi) > 0:
            for i in range(len(self.current_roi)):
                save_x, save_y = self.current_roi[i]
                disp_x, disp_y = self.coords_for_display(save_x, save_y)
                cv2.circle(self.display_frame, (disp_x, disp_y), 3, 
                          self.roi_color_map[self.current_region_type], -1)
                if i > 0:
                    prev_save_x, prev_save_y = self.current_roi[i-1]
                    prev_disp_x, prev_disp_y = self.coords_for_display(prev_save_x, prev_save_y)
                    cv2.line(self.display_frame, (prev_disp_x, prev_disp_y), (disp_x, disp_y), 
                            self.roi_color_map[self.current_region_type], 2)
        
        cv2.imshow('Region Selector', self.display_frame)
    
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
    
    def save_region_info(self):
        """Save region information to a file"""
        if self.output_path:
            # Save region info to JSON file
            import json
            
            region_info = {
                'video_info': {
                    'original_dimensions': [self.original_width, self.original_height],
                    'display_dimensions': [self.display_width, self.display_height],
                    'scale_factors': [self.scale_x, self.scale_y],
                    'coordinate_system': 'resized' if self.use_resized_coords else 'original'
                },
                'regions': self.regions
            }
            
            info_file = os.path.splitext(self.output_path)[0] + "_regions.json"
            with open(info_file, 'w') as f:
                json.dump(region_info, f, indent=2)
            print(f"Region info saved to {info_file}")
    
    def run(self):
        """Run the region selector tool"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return False
        
        # Get video properties
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate resize parameters
        self.calculate_resize_parameters(original_width, original_height)
        
        # Read the first frame
        ret, self.original_frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            return False
        
        # Create display frame (potentially resized)
        self.display_frame = self.resize_frame_for_display(self.original_frame)
        
        # Set up the window and mouse callback
        cv2.namedWindow('Region Selector')
        cv2.setMouseCallback('Region Selector', self.mouse_callback)
        
        coords_type = "Resized" if self.use_resized_coords else "Original"
        coords_dims = f"{self.display_width}x{self.display_height}" if self.use_resized_coords else f"{self.original_width}x{self.original_height}"
        
        print(f"\nRegion Selector Tool (with Video Resize Support)")
        print("=" * 50)
        print("Instructions:")
        print("  - Press 1: Select stop line (click two points to define a line)")
        print("  - Press 2: Select green zone (click multiple points, then press 'c' to close polygon)")
        print("  - Press 3: Select red zone (click multiple points, then press 'c' to close polygon)")
        print("  - Press 4: Select traffic light bounding box (click and drag)")
        print("  - Press 'c': Complete current polygon")
        print("  - Press 'r': Reset current region")
        print("  - Press 's': Save all regions and show command")
        print("  - Press 'q': Quit without saving")
        print(f"\nNOTE: Coordinates will be saved as {coords_type} coordinates ({coords_dims}).")
        print("=" * 50 + "\n")
        
        while True:
            cv2.imshow('Region Selector', self.display_frame)
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
                    coords_type = "Resized" if self.use_resized_coords else "Original"
                    print(f"{self.current_region_type} coordinates: {self.regions[self.current_region_type]} [{coords_type} coords]")
                
            elif key == ord('s'):
                # Generate command and save frame if desired
                if self.output_path:
                    cv2.imwrite(self.output_path, self.display_frame)
                    print(f"Saved annotated frame to {self.output_path}")
                    self.save_region_info()
                
                # Generate command string
                cmd = "python traffic_violation_detector.py"
                
                for region_type in ['stop_line', 'green_zone', 'red_zone', 'traffic_light_bbox']:
                    coords = self.format_coordinates(region_type)
                    if coords:
                        cmd += f" --{region_type} {coords}"
                
                # Add resize parameters if used
                if self.use_resized_coords:
                    # If using resized coords, need to pass resize parameters to the detector
                    if self.scale_factor:
                        cmd += f" --scale_factor {self.scale_factor}"
                    elif self.target_width and self.target_height:
                        cmd += f" --target_width {self.target_width} --target_height {self.target_height}"
                    elif self.target_width:
                        cmd += f" --target_width {self.target_width}"
                    elif self.target_height:
                        cmd += f" --target_height {self.target_height}"
                
                # Add default arguments
                cmd += " --input <VIDEO_PATH> --output_video result_violation.mp4 --output_data violations_data.csv"
                cmd += " --detection_method combined --min_frames_in_green 5 --trajectory_interpolation 15"
                cmd += " --tolerance 20 --save_preview_frames 10"
                
                coords_type = "Resized" if self.use_resized_coords else "Original"
                coords_dims = f"{self.display_width}x{self.display_height}" if self.use_resized_coords else f"{self.original_width}x{self.original_height}"
                
                print("\nGenerated Command:")
                print("=" * 80)
                print(cmd)
                print("=" * 80)
                print(f"\nCoordinates are in {coords_type} format ({coords_dims}).")
                if self.use_resized_coords:
                    print("IMPORTANT: Use the same resize parameters when running the detector!")
                else:
                    print("Coordinates will work correctly with any resize settings in the detector.")
                
                if self.output_path:
                    # Save the command to a text file
                    cmd_file = os.path.splitext(self.output_path)[0] + "_command.txt"
                    with open(cmd_file, 'w') as f:
                        f.write(cmd)
                        f.write(f"\n\n# Video Information:\n")
                        f.write(f"# Original dimensions: {self.original_width}x{self.original_height}\n")
                        f.write(f"# Display dimensions: {self.display_width}x{self.display_height}\n")
                        f.write(f"# Scale factors: x={self.scale_x:.3f}, y={self.scale_y:.3f}\n")
                        f.write(f"# Coordinate system: {coords_type} ({coords_dims})\n")
                    print(f"Command also saved to {cmd_file}")
        
        cap.release()
        cv2.destroyAllWindows()
        return True


def main():
    parser = argparse.ArgumentParser(description="Select regions for traffic violation detection with video resize support")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to the input video")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the annotated frame (optional)")
    
    # Resize options
    resize_group = parser.add_mutually_exclusive_group()
    resize_group.add_argument("--scale_factor", type=float, default=None,
                             help="Scale factor for resizing (e.g., 0.5 for half size)")
    resize_group.add_argument("--target_width", type=int, default=None,
                             help="Target width for resizing (height will be calculated to maintain aspect ratio)")
    parser.add_argument("--target_height", type=int, default=None,
                        help="Target height for resizing (can be used with target_width for specific dimensions)")
    
    # NEW: Option to save coordinates in resized format
    parser.add_argument("--use_resized_coords", action="store_true", default=False,
                        help="Save coordinates in resized format instead of original format")
    
    args = parser.parse_args()
    
    # Validate resize arguments
    if args.target_height and not args.target_width:
        # If only height is specified, use it as the primary dimension
        args.target_width = None
    
    selector = RegionSelector(
        video_path=args.input, 
        output_path=args.output,
        target_width=args.target_width,
        target_height=args.target_height,
        scale_factor=args.scale_factor,
        use_resized_coords=args.use_resized_coords
    )
    selector.run()


if __name__ == "__main__":
    main()