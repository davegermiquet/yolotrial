import cv2
import pafy
import time
import numpy as np
import argparse
import os
import json
from datetime import datetime
from vidgear.gears import CamGear
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class VehicleTracker:
    def __init__(self, source_path, model_path="yolo11n.pt", region_file=None):
        """
        Initialize the vehicle tracker with source video and model
        
        Args:
            source_path (str): Path to video file, camera, or YouTube URL
            model_path (str): Path to YOLO model
            region_file (str): Optional path to saved regions file
        """
        self.source_path = source_path
        self.model_path = model_path
        self.regions = []  # List to store defined regions
        self.track_data = {}  # Dictionary to store tracking data
        self.tracked_vehicles = {}  # Dictionary to store vehicle tracking info
        self.current_region_points = []  # Temporary storage for region points
        self.fps = 0
        self.results_log = []  # Store speed measurement results
        
        # Set up tracker
        self.tracker = DeepSort(
            max_age=60,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.2,
            nn_budget=None,
            gating_only_position=False,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None,
        )
        
        # Load the YOLO model
        self.model = YOLO(model_path)
        
        # Object classes of interest (car, truck, motorcycle)
        self.target_classes = [2, 3, 7]
        
        # Color palette for visualization
        self.colors = {
            'region': (0, 255, 0),      # Green for regions
            'active_region': (0, 255, 255),  # Yellow for active region
            'detection': (255, 0, 0),   # Blue for detections
            'tracking': (0, 0, 255),    # Red for tracking
            'speed_text': (255, 255, 0) # Yellow for speed text
        }
        
        # Load regions if file exists
        if region_file and os.path.exists(region_file):
            self.load_regions(region_file)
            
    def load_regions(self, region_file):
        """Load previously saved regions from a file"""
        try:
            with open(region_file, 'r') as f:
                data = json.load(f)
                self.regions = data['regions']
                print(f"Loaded {len(self.regions)} regions from {region_file}")
        except Exception as e:
            print(f"Error loading regions: {e}")
            
    def save_regions(self, region_file):
        """Save defined regions to a file"""
        try:
            with open(region_file, 'w') as f:
                json.dump({'regions': self.regions}, f)
            print(f"Saved {len(self.regions)} regions to {region_file}")
        except Exception as e:
            print(f"Error saving regions: {e}")
            
    def open_video_source(self):
        """Open the video source based on the type (file, camera, or YouTube URL)"""
        if self.source_path.startswith('http') and 'youtube' in self.source_path:
            # YouTube URL
            try:
                video = pafy.new(self.source_path)
                best = video.getbest(preftype="mp4")
                options = {"STREAM_RESOLUTION": "720p"}
                self.cap = CamGear(
                    source=self.source_path, stream_mode=True, logging=True, **options
                ).start()
                self.fps = 30  # Assume 30fps for YouTube streams
                return True
            except Exception as e:
                print(f"Error opening YouTube stream: {e}")
                return False
        else:
            # Local file or camera
            self.cap = cv2.VideoCapture(self.source_path)
            if not self.cap.isOpened():
                print(f"Error: Could not open video source {self.source_path}")
                return False
            
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30  # Use 30fps as fallback
            
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video opened: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
            return True
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for region definition"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_region_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to cancel the last point
            if self.current_region_points:
                self.current_region_points.pop()
        elif event == cv2.EVENT_MBUTTONDOWN:
            # Middle click to complete current region
            if len(self.current_region_points) >= 2:
                # Create a region from the current points
                region = {
                    'id': len(self.regions),
                    'name': f"Region {len(self.regions)+1}",
                    'points': self.current_region_points.copy(),
                    'type': 'checkpoint',  # 'checkpoint' or 'zone'
                    'direction': 'both'    # 'entry', 'exit', or 'both'
                }
                self.regions.append(region)
                self.current_region_points = []
                print(f"Region {region['name']} created")
                
    def calculate_speed(self, track_id, region1_time, region2_time, distance_meters):
        """
        Calculate the speed of a vehicle between two checkpoint regions
        
        Args:
            track_id: The ID of the tracked vehicle
            region1_time: Timestamp when vehicle entered first region
            region2_time: Timestamp when vehicle entered second region
            distance_meters: Distance between regions in meters
            
        Returns:
            float: Speed in km/h
        """
        if region2_time <= region1_time:
            return 0
        
        time_diff = region2_time - region1_time  # Time in seconds
        
        if time_diff == 0:
            return 0
            
        # Calculate speed in meters per second
        speed_mps = distance_meters / time_diff
        
        # Convert to km/h
        speed_kmh = speed_mps * 3.6
        
        # Log the result
        result = {
            'track_id': track_id,
            'region1_time': region1_time,
            'region2_time': region2_time,
            'time_diff': time_diff,
            'distance_meters': distance_meters,
            'speed_kmh': speed_kmh,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.results_log.append(result)
        
        return speed_kmh
        
    def is_point_in_region(self, point, region):
        """Check if a point is inside a region"""
        x, y = point
        polygon = np.array(region['points'], np.int32)
        polygon = polygon.reshape((-1, 1, 2))
        result = cv2.pointPolygon(np.array([[x, y]], dtype=np.float32), polygon)
        return result > 0
        
    def is_box_in_region(self, box, region):
        """Check if a bounding box intersects with a region"""
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        bottom_y = y2  # Bottom center of the box
        
        return self.is_point_in_region((center_x, bottom_y), region)
        
    def draw_regions(self, frame):
        """Draw all defined regions on the frame"""
        # Draw current region being defined
        if len(self.current_region_points) > 0:
            for i in range(len(self.current_region_points)):
                cv2.circle(frame, self.current_region_points[i], 5, self.colors['active_region'], -1)
                
                if i > 0:
                    cv2.line(frame, 
                             self.current_region_points[i-1], 
                             self.current_region_points[i], 
                             self.colors['active_region'], 2)
                    
            # Draw line from last point to first if there are more than 2 points
            if len(self.current_region_points) > 2:
                cv2.line(frame, 
                         self.current_region_points[-1], 
                         self.current_region_points[0], 
                         self.colors['active_region'], 2, cv2.LINE_DASH)
        
        # Draw all saved regions
        for region in self.regions:
            points = np.array(region['points'], np.int32)
            points = points.reshape((-1, 1, 2))
            
            cv2.polylines(frame, [points], True, self.colors['region'], 2)
            
            # Draw region name
            centroid_x = sum(p[0] for p in region['points']) // len(region['points'])
            centroid_y = sum(p[1] for p in region['points']) // len(region['points'])
            
            cv2.putText(frame, region['name'], (centroid_x, centroid_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['region'], 2)
                        
    def process_frame(self, frame, frame_count):
        """Process a single frame for detection and tracking"""
        if frame is None:
            return frame
            
        # Run YOLO detection on the frame
        results = self.model(frame)
        detection_list = []
        
        # Process detection results
        for r in results:
            for box_data in r.boxes.data.tolist():
                class_id = int(box_data[5])
                
                # Filter classes (car, truck, motorcycle)
                if class_id not in self.target_classes:
                    continue
                    
                confidence = float(box_data[4])
                
                # Filter low confidence detections
                if confidence < 0.3:
                    continue
                    
                # Extract bounding box coordinates
                xmin, ymin, xmax, ymax = int(box_data[0]), int(box_data[1]), int(box_data[2]), int(box_data[3])
                
                # Format for DeepSort tracker
                detection_list.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])
                
                # Draw detection box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), self.colors['detection'], 1)
        
        # Update tracks with DeepSort
        tracks = self.tracker.update_tracks(detection_list, frame=frame)
        
        # Process each tracked object
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            # Get tracking ID
            track_id = track.track_id
            
            # Convert to bounding box format (x1, y1, x2, y2)
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw tracking box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['tracking'], 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['tracking'], 2)
            
            # Initialize track data if this is a new ID
            if track_id not in self.tracked_vehicles:
                self.tracked_vehicles[track_id] = {
                    'first_seen': frame_count,
                    'last_seen': frame_count,
                    'regions': {},  # Store timing info for each region
                    'speeds': []    # Store calculated speeds
                }
            
            # Update track data
            self.tracked_vehicles[track_id]['last_seen'] = frame_count
            
            # Check for region interactions
            for region in self.regions:
                region_id = region['id']
                
                # Check if vehicle is in this region
                if self.is_box_in_region(bbox, region):
                    # Record entry time if not already recorded
                    if region_id not in self.tracked_vehicles[track_id]['regions']:
                        current_time = time.time()
                        self.tracked_vehicles[track_id]['regions'][region_id] = current_time
                        
                        # If we have timing data for two regions, calculate speed
                        if len(self.tracked_vehicles[track_id]['regions']) >= 2:
                            # Find the two regions with the earliest timestamps
                            region_timestamps = sorted(self.tracked_vehicles[track_id]['regions'].items(), 
                                                      key=lambda x: x[1])
                            
                            # Get the first two regions (sorted by timestamp)
                            r1_id, r1_time = region_timestamps[0]
                            r2_id, r2_time = region_timestamps[1]
                            
                            # Calculate speed (assuming 10 meters between regions for now)
                            # In a real implementation, you would configure the actual distance
                            distance_meters = 10.0  # Example value - should be configured
                            
                            speed = self.calculate_speed(track_id, r1_time, r2_time, distance_meters)
                            
                            if speed > 0:
                                self.tracked_vehicles[track_id]['speeds'].append(speed)
                                
                                # Show speed on frame
                                avg_speed = sum(self.tracked_vehicles[track_id]['speeds']) / len(self.tracked_vehicles[track_id]['speeds'])
                                speed_text = f"{avg_speed:.1f} km/h"
                                
                                cv2.putText(frame, speed_text, (x1, y2+20), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['speed_text'], 2)
        
        # Draw regions on the frame
        self.draw_regions(frame)
        
        # Add info text
        cv2.putText(frame, f"Tracked vehicles: {len(self.tracked_vehicles)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                   
        # Instructions
        cv2.putText(frame, "Left-click: Add point | Right-click: Remove point | Middle-click: Complete region", 
                   (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
        
    def save_results(self, output_file):
        """Save speed measurement results to a CSV file"""
        if not self.results_log:
            print("No results to save")
            return
            
        try:
            with open(output_file, 'w') as f:
                # Write header
                f.write("track_id,region1_time,region2_time,time_diff,distance_meters,speed_kmh,timestamp\n")
                
                # Write data
                for result in self.results_log:
                    f.write(f"{result['track_id']},{result['region1_time']},{result['region2_time']},"
                           f"{result['time_diff']},{result['distance_meters']},{result['speed_kmh']},"
                           f"{result['timestamp']}\n")
                           
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
            
    def run(self):
        """Main processing loop"""
        if not self.open_video_source():
            print("Failed to open video source")
            return
            
        # Create window and set mouse callback
        cv2.namedWindow("Vehicle Tracker")
        cv2.setMouseCallback("Vehicle Tracker", self.mouse_callback)
        
        frame_count = 0
        skip_frames = 2  # Process every n frames for performance
        
        # For FPS calculation
        start_time = time.time()
        processed_frames = 0
        
        # Main loop
        while True:
            # Read frame
            if isinstance(self.cap, cv2.VideoCapture):
                ret, frame = self.cap.read()
                if not ret:
                    break
            else:  # CamGear for YouTube
                frame = self.cap.read()
                if frame is None:
                    break
                    
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % (skip_frames + 1) != 0:
                continue
                
            processed_frames += 1
            
            # Process the frame
            processed_frame = self.process_frame(frame, frame_count)
            
            # Display the frame
            cv2.imshow("Vehicle Tracker", processed_frame)
            
            # Calculate and show FPS every second
            if time.time() - start_time >= 1.0:
                actual_fps = processed_frames / (time.time() - start_time)
                processed_frames = 0
                start_time = time.time()
                print(f"Processing FPS: {actual_fps:.2f}")
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                break
            elif key == ord('s'):  # Save regions
                self.save_regions("regions.json")
            elif key == ord('c'):  # Clear current region points
                self.current_region_points = []
            elif key == ord('d'):  # Delete last region
                if self.regions:
                    self.regions.pop()
                    
        # Clean up
        if isinstance(self.cap, cv2.VideoCapture):
            self.cap.release()
        else:
            self.cap.stop()
            
        cv2.destroyAllWindows()
        
        # Save results
        self.save_results("speed_results.csv")
        

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Vehicle Tracking and Speed Estimation")
    parser.add_argument('--source', type=str, default="/dev/video0", 
                       help="Path to video file, camera index, or YouTube URL")
    parser.add_argument('--model', type=str, default="yolo11n.pt", 
                       help="Path to YOLO model")
    parser.add_argument('--regions', type=str, default="regions.json", 
                       help="Path to regions file")
    
    args = parser.parse_args()
    
    # Create and run tracker
    tracker = VehicleTracker(
        source_path=args.source,
        model_path=args.model,
        region_file=args.regions
    )
    
    tracker.run()