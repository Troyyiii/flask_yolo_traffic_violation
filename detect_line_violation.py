import os
import cv2
import logging
import cvzone
import numpy as np
from datetime import datetime
from .sort import Sort
from .model import load_model

logger = logging.getLogger(__name__)

class DetectLineViolation:
    def __init__(self, file_dir):
        # Load custom trained model
        self.line_model = load_model("./model/line_test_best100.pt")
        self.crosswalk_model = load_model("./model/yolov5_crosswalk_best50.pt")
        
        # Initialize tracker
        self.tracker = Sort(max_age=180, min_hits=3, iou_threshold=0.3)
        
        # Initialize area for boundary detection
        self.area = []
        
        # Initialize trails dictionary
        self.trails = {}
        
        # initialize crosswalk direction check flag
        self.crosswalk_dir_check = False
        
        # initialize traffic light status
        self.traffic_light_status = "Unknown"
        
        # Initialize list of violator ID & counter
        self.traffic_light_violator_list = []
        self.traffic_light_violator_counter = 0
        self.wrong_way_violator_list = []
        self.wrong_way_violator_counter = 0
        
        # Initialize list of clear vehicle
        self.traffic_light_clear_list = []
        
        self.file_dir = file_dir
        self.create_folder()
        
    def create_folder(self):
        self.traffic_violation_dir = os.path.join(self.file_dir, 'traffic_violation')
        os.makedirs(self.traffic_violation_dir, exist_ok=True)
        
        self.traffic_line_violation_dir = os.path.join(self.traffic_violation_dir, 'traffic_line')
        self.wrong_way_violation_dir = os.path.join(self.traffic_violation_dir, 'wrong_way')
        
        os.makedirs(self.traffic_line_violation_dir, exist_ok=True)
        os.makedirs(self.wrong_way_violation_dir, exist_ok=True)
        
    def start_detect(self, frame):
        if not self.area:
            logger.info("=== Detecting crosswalk boundary ===")
            
            trapezoid_frame = self.crop_to_trapezoid(frame.copy())
            processed_frame = self.check_crosswalk(trapezoid_frame, frame.copy())
            
            cvzone.putTextRect(processed_frame, "Detecting Crosswalk Boundary...", (10, 10), scale=1, thickness=1)
            
            return processed_frame
        else:
            processed_frame = self.detect_object(frame.copy())
            return processed_frame
    
    def crop_to_trapezoid(self, frame):
        height, width = frame.shape[:2]
        region_of_interest_vertices = [
            (0, height),
            (width / 7, height / 2),
            (6 * width / 7, height / 2),
            (width, height)
        ]
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        cv2.fillPoly(mask, np.array([region_of_interest_vertices], np.int32), 255)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        return masked_frame
    
    def check_crosswalk(self, crop_frame, real_frame):
        results = self.line_model(crop_frame)
        objects = results.pandas().xyxy[0]
        processed_frame = results.render()[0]
        
        logger.info(f"Detected objects : \n{objects}\n")
        
        if objects is None or len(objects) == 0:
            logger.info("=== Road is clear ===")
            processed_frame = self.detect_crosswalk(real_frame)
        
        return processed_frame
    
    def detect_crosswalk(self, frame):
        results = self.crosswalk_model(frame.copy())
        objects = results.pandas().xyxy[0]
        
        for _, row in objects.iterrows():
            xmin = int(row["xmin"])
            ymin = int(row["ymin"])
            xmax = int(row["xmax"])
            ymax = int(row["ymax"])
            confidence = float(f"{row['confidence']:.2f}")
            class_id = int(row["class"])
            name = row["name"]
            
            if confidence >= 0.5:
                cy = (ymin + ymax) // 2
                
                self.area.append({
                    "coords": np.array([
                        [xmin, cy],
                        [xmax, cy]
                    ], dtype=np.int32),
                    "north_count": 0,
                    "south_count": 0,
                    "status_dir": "Undefined",
                    "counted_idx": set()
                })
                
                logger.info(f"Class: ({class_id}){name}")
                logger.info(f"Confidence: {confidence}")
                logger.info(f"\nDetected Crosswalk: \n{self.area}\n")
                
                cv2.circle(frame, (xmin, cy), 5, (0, 0, 255), -1)
                cv2.circle(frame, (xmax, cy), 5, (0, 0, 255), -1)
                
        return frame
    
    def detect_object(self, frame):
        # line model detection
        line_results = self.line_model(frame.copy())
        line_objects = line_results.pandas().xyxy[0]
        
        detections = self.set_tracker(line_objects)
        tracker_results = self.tracker.update(detections)
        
        logger.info(f"Detection : \n{detections}\n")
        logger.info(f"Tracker Results : \n{tracker_results}\n")
        
        # check traffic light status
        frame = self.check_traffic_light_status(line_objects, frame)
        
        processed_frame = self.draw_bounding_box(frame, tracker_results)
        
        return processed_frame
    
    def set_tracker(self, objects):
        detections = np.empty((0, 5)) # Initialize empty array
        
        for _, row in objects.iterrows():
            xmin = int(row["xmin"])
            ymin = int(row["ymin"])
            xmax = int(row["xmax"])
            ymax = int(row["ymax"])
            confidence = float(f"{row['confidence']:.2f}")
            class_id = int(row["class"])
            name = row["name"]
            
            if confidence >= 0.5 and (class_id == 0 or class_id == 1):
                logger.info(f"Class: ({class_id}){name}")
                logger.info(f"Confidence: {confidence}\n")
                
                coordinate_list = np.array([xmin, ymin, xmax, ymax, confidence])
                detections = np.vstack([detections, coordinate_list])
        
        return detections
    
    def check_traffic_light_status(self, objects, frame):
        if self.crosswalk_dir_check:
            green_count, red_count, green_confidence_sum, red_confidence_sum, frame = self.count_traffic_lights(objects, frame)
            self.update_traffic_light_status(green_count, red_count, green_confidence_sum, red_confidence_sum)
            logger.info(f"Traffic Light Status: {self.traffic_light_status}")
            cvzone.putTextRect(frame, f"G: {green_count} {green_confidence_sum}, R: {red_count} {red_confidence_sum}", (25, 100), scale=1, thickness=1, offset=3)

        return frame
    
    def count_traffic_lights(self, objects, frame):
        green_count = 0
        red_count = 0
        green_confidence_sum = 0.0
        red_confidence_sum = 0.0

        for _, row in objects.iterrows():
            xmin = int(row["xmin"])
            ymin = int(row["ymin"])
            xmax = int(row["xmax"])
            ymax = int(row["ymax"])
            confidence = float(f"{row['confidence']:.2f}")
            class_id = int(row["class"])
            name = row["name"]

            if confidence >= 0.75:
                if class_id == 2:
                    green_count += 1
                    green_confidence_sum += confidence
                    logger.info("Green light detected")
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f"{name}", (max(0, xmin), max(35, ymin)), scale=0.8, thickness=1, offset=3)
                if class_id == 3:
                    red_count += 1
                    red_confidence_sum += confidence
                    logger.info("Red light detected")
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    cvzone.putTextRect(frame, f"{name}", (max(0, xmin), max(35, ymin)), scale=0.8, thickness=1, offset=3)

        return green_count, red_count, green_confidence_sum, red_confidence_sum, frame
    
    def update_traffic_light_status(self, green_count, red_count, green_confidence_sum, red_confidence_sum):
        if green_count > 0 or red_count > 0:
            if green_count > red_count:
                self.traffic_light_status = "Green"
            elif red_count > green_count:
                self.traffic_light_status = "Red"
            else:
                green_avg_confidence = green_confidence_sum / green_count
                red_avg_confidence = red_confidence_sum / red_count

                if green_avg_confidence > red_avg_confidence:
                    self.traffic_light_status = "Green"
                elif red_avg_confidence > green_avg_confidence:
                    self.traffic_light_status = "Red"
                else:
                    self.traffic_light_status = "Unknown"

                logger.info(f"Traffic detected same value, count the average confidence. Results: Green: {green_avg_confidence}, Red: {red_avg_confidence}")
        else:
            self.traffic_light_status = "Unknown"
    
    def draw_bounding_box(self, frame, tracker_results):
        for result in tracker_results:
            xmin, ymin, xmax, ymax, idx = map(int, result)
            
            # calculate center of bounding box
            cx = int(xmin + xmax) // 2
            cy = int(ymin + ymax) // 2
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)
            cvzone.putTextRect(frame, f"{idx}", (max(0, xmin), max(35, ymin)), scale=0.8, thickness=1, offset=3)
            
            self.update_trails(idx, cx, ymax)
            
            # draw trails
            if idx in self.trails:
                for i in range(1, len(self.trails[idx])):
                    thickness = int(np.sqrt(64 / float(len(self.trails[idx]) - i)) * 1.5)
                    thickness = max(1, min(thickness, 10))
                    
                    cv2.line(frame, self.trails[idx][i-1], self.trails[idx][i], (255, 0, 0), thickness)
                    
                    # get object direction
                    obj_dir = self.get_direction(self.trails[idx][i-1], self.trails[idx][i])
                    
                    cvzone.putTextRect(frame, f"{obj_dir}", (xmax, ymin), scale=0.8, thickness=1, offset=3)
                    
                    if self.crosswalk_dir_check is False:
                        for area in self.area:
                            if area["status_dir"] == "Undefined":
                                if self.do_lines_intersect(self.trails[idx][i-1], self.trails[idx][i], area["coords"][0], area["coords"][1]):
                                    if idx not in area["counted_idx"]:
                                        if obj_dir == "North":
                                            area["north_count"] += 1
                                        elif obj_dir == "South":
                                            area["south_count"] += 1
                                        
                                        area["counted_idx"].add(idx)
                                        
                                        if area["north_count"] >= 5:
                                            area["status_dir"] = "North"
                                        elif area["south_count"] >= 5:
                                            area["status_dir"] = "South"
                                        
                                        logger.info(f"Area: {area['coords']}, North Count: {area['north_count']}, South Count: {area['south_count']}, Status: {area['status_dir']}")

                        if all(area["status_dir"] != "Undefined" for area in self.area):
                            self.crosswalk_dir_check = True
                            logger.info("All areas have defined directions.")
                    
                    if self.crosswalk_dir_check:
                        for area in self.area:
                            # check traffic light violation
                            if area["status_dir"] == "North":
                                if self.do_lines_intersect(self.trails[idx][i-1], self.trails[idx][i], area["coords"][0], area["coords"][1]):
                                    if self.traffic_light_status == "Red":
                                        if idx not in self.traffic_light_violator_list and idx not in self.traffic_light_clear_list:
                                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                                            cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
                                            self.traffic_light_violator_list.append(idx)
                                            self.traffic_light_violator_counter += 1
                                            self.capture_violation(frame.copy(), (xmin, ymin, xmax, ymax), "traffic_line")
                                            logger.info(f"Violator detected! ID: {idx}\nTotal Violator: {self.traffic_light_violator_counter}\nViolator list: {self.traffic_light_violator_list}\n")
                                    if self.traffic_light_status == "Green":
                                        if idx not in self.traffic_light_clear_list and idx not in self.traffic_light_violator_list:
                                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                            cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
                                            self.traffic_light_clear_list.append(idx)
                            
                            # check wrong way violation
                            if obj_dir == "North":
                                if area["status_dir"] == "South":
                                    if self.do_lines_intersect(self.trails[idx][i-1], self.trails[idx][i], area["coords"][0], area["coords"][1]):
                                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                                        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
                                        if idx not in self.wrong_way_violator_list:
                                            self.wrong_way_violator_list.append(idx)
                                            self.wrong_way_violator_counter += 1
                                            self.capture_violation(frame.copy(), (xmin, ymin, xmax, ymax), "wrong_way")
                                            logger.info("South line violated!")
                                            logger.info(f"Wrong way violator detected! ID: {idx}\nTotal Violator: {self.wrong_way_violator_counter}\nViolator list: {self.wrong_way_violator_list}\n")
                            
                            if obj_dir == "South":
                                if area["status_dir"] == "North":
                                    if self.do_lines_intersect(self.trails[idx][i-1], self.trails[idx][i], area["coords"][0], area["coords"][1]):
                                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                                        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
                                        if idx not in self.wrong_way_violator_list:
                                            self.wrong_way_violator_list.append(idx)
                                            self.wrong_way_violator_counter += 1
                                            self.capture_violation(frame.copy(), (xmin, ymin, xmax, ymax), "wrong_way")
                                            logger.info("North line violated!")
                                            logger.info(f"Wrong way violator detected! ID: {idx}\nTotal Violator: {self.wrong_way_violator_counter}\nViolator list: {self.wrong_way_violator_list}\n")
            
        return frame
    
    def update_trails(self, idx, cx, ymax):
        if idx not in self.trails:
            self.trails[idx] = []
        self.trails[idx].append((cx, ymax))
        
    def get_direction(self, p1, p2):
        direction = ""
        
        if p1[1] > p2[1]:
            direction = "North"
        elif p1[1] < p2[1]:
            direction = "South"
        
        return direction
    
    def do_lines_intersect(self, p1, p2, q1, q2):
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)
    
    def capture_violation(self, frame, bbox, violation, padding = 250):
        rxmin, rymin, rxmax, rymax = bbox
        xmin = max(rxmin - padding, 0)
        ymin = max(rymin - padding, 0)
        xmax = min(rxmax + padding, frame.shape[1])
        ymax = min(rymax + padding, frame.shape[0])
        
        timestamp = datetime.now().strftime("%H.%M.%S")
        
        if violation == "traffic_line":
            violation_directory = self.traffic_line_violation_dir
            image_filename = f"{timestamp}_{self.traffic_light_violator_counter}.jpg"
        if violation == "wrong_way":
            violation_directory = self.wrong_way_violation_dir
            image_filename = f"{timestamp}_{self.wrong_way_violator_counter}.jpg"
            
        image_output_path = os.path.join(violation_directory, image_filename)
        cropped_frame = frame[ymin:ymax, xmin:xmax]
        cv2.imwrite(image_output_path, cropped_frame)
        
        logger.info("Violation captured!")