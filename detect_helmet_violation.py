import os
import cv2
import logging
import cvzone
import numpy as np
from datetime import datetime
from sort import Sort
from model import load_model

logger = logging.getLogger(__name__)

class DetectHelmetViolation:
    def __init__(self, file_dir):
        # Load custom trained model
        self.helmet_model = load_model("./model/helm_test_best50.pt")
        
        # Initialize tracker
        self.tracker = Sort(max_age=180, min_hits=3, iou_threshold=0.3)
        
        # Initialize list of violator ID & counter
        self.helmet_violation_counter = 0
        self.helmet_violator_id_list = []
        
        self.file_dir = file_dir
        self.create_folder()
    
    def create_folder(self):
        self.traffic_violation_dir = os.path.join(self.file_dir, 'traffic_violation')
        os.makedirs(self.traffic_violation_dir, exist_ok=True)
        self.helmet_violation_dir = os.path.join(self.traffic_violation_dir, 'helmet')
        os.makedirs(self.helmet_violation_dir, exist_ok=True)
    
    def start_detect(self, frame):
        processed_frame = self.detect_object(frame.copy())
        return processed_frame
    
    def detect_object(self, frame):
        results = self.helmet_model(frame.copy())
        objects = results.pandas().xyxy[0]
        
        # Use YOLO bounding box
        # yolo_processed_frame = results.render()[0]
        
        rider_detections, no_helmet_detections = self.get_detections(objects)
        tracker_results = self.tracker.update(rider_detections)
        
        logger.info(f"Rider Detection : \n{rider_detections}\n")
        logger.info(f"No-helm Detection : \n{no_helmet_detections}\n")
        logger.info(f"Tracker Results : \n{tracker_results}\n")
        
        processed_frame = self.draw_bounding_box(frame.copy(), tracker_results)
        processed_frame = self.check_helmet_violation(processed_frame, tracker_results, no_helmet_detections)
        
        return processed_frame

    def get_detections(self, objects):
        rider_detections = np.empty((0, 5))
        no_helmet_detections = np.empty((0, 5))
        
        for _, row in objects.iterrows():
            xmin = int(row["xmin"])
            ymin = int(row["ymin"])
            xmax = int(row["xmax"])
            ymax = int(row["ymax"])
            confidence = float(f"{row['confidence']:.2f}")
            class_id = int(row["class"])
            name = row["name"]
            
            if confidence >= 0.7 and class_id == 2:
                logger.info(f"Class: ({class_id}){name}")
                logger.info(f"Confidence: {confidence}\n")
                
                coordinate_list = np.array([xmin, ymin, xmax, ymax, confidence])
                rider_detections = np.vstack([rider_detections, coordinate_list])
            
            if confidence >= 0.8 and class_id == 1:
                logger.info(f"Class: ({class_id}){name}")
                logger.info(f"Confidence: {confidence}\n")
                
                coordinate_list = np.array([xmin, ymin, xmax, ymax, confidence])
                no_helmet_detections = np.vstack([no_helmet_detections, coordinate_list])
        
        return rider_detections, no_helmet_detections

    def draw_bounding_box(self, frame, tracker_results):
        for result in tracker_results:
            xmin, ymin, xmax, ymax, idx = map(int, result)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cvzone.putTextRect(frame, f"{idx}", (max(0, xmin), max(35, ymin)), scale=0.8, thickness=1, offset=3)
        
        return frame
    
    def check_helmet_violation(self, frame, tracker_results, no_helmet_detections):
        for detection in no_helmet_detections:
            xmin, ymin, xmax, ymax, _ = map(int, detection)
            cx = int(xmin + xmax) // 2
            cy = int(ymin + ymax) // 2
            
            for result in tracker_results:
                rxmin, rymin, rxmax, rymax, idx = map(int, result)
                if rxmin <= cx <= rxmax and rymin <= cy <= rymax:
                    if idx not in self.helmet_violator_id_list:
                        self.helmet_violation_counter += 1
                        self.helmet_violator_id_list.append(idx)
                        cv2.rectangle(frame, (rxmin, rymin), (rxmax, rymax), (0, 0, 255), 2)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                        self.capture_violation(frame.copy(), (rxmin, rymin, rxmax, rymax))
                        logger.info(f"Helmet violation detected! Rider ID: {idx}\nTotal Violations: {self.helmet_violation_counter}\nViolator list: {self.helmet_violator_id_list}\n")
    
            
        return frame
    
    def capture_violation(self, frame, bbox, padding = 20):
        rxmin, rymin, rxmax, rymax = bbox
        xmin = max(rxmin - padding, 0)
        ymin = max(rymin - padding, 0)
        xmax = min(rxmax + padding, frame.shape[1])
        ymax = min(rymax + padding, frame.shape[0])
        
        timestamp = datetime.now().strftime("%H.%M.%S")
        
        violation_directory = self.helmet_violation_dir
        image_filename = f"{timestamp}_{self.helmet_violation_counter}.jpg"
            
        image_output_path = os.path.join(violation_directory, image_filename)
        cropped_frame = frame[ymin:ymax, xmin:xmax]
        cv2.imwrite(image_output_path, cropped_frame)
        
        logger.info("Violation captured!")