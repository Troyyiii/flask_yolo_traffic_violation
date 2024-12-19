import os
import cv2
import logging
import cvzone
from detect_line_violation import DetectLineViolation
from detect_helmet_violation import DetectHelmetViolation

logger = logging.getLogger(__name__)

def start_detection(file_dir, video_input_path, violation_type):
    capture = cv2.VideoCapture(video_input_path)
    
    fps = capture.get(cv2.CAP_PROP_FPS)
    
    video_id = os.path.basename(file_dir)
    
    result_dir = os.path.join(file_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)
    
    if violation_type == "line":
        output_file_path = os.path.join(result_dir, f"{video_id}_line_result.mp4")
        detect_violation = DetectLineViolation(file_dir)
    if violation_type == "helmet":
        output_file_path = os.path.join(result_dir, f"{video_id}_helmet_result.mp4")
        detect_violation = DetectHelmetViolation(file_dir)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (1280, 720))
        
    while True:
        ret, frame = capture.read()
        
        if not ret:
            logger.error("Failed to open video file")
            break
        
        frame = cv2.resize(frame, (1280, 720))
        
        if violation_type == "line":
            processed_frame = detect_violation.start_detect(frame)
            draw_detected_areas(processed_frame, detect_violation.area)
            if detect_violation.crosswalk_dir_check:
                if detect_violation.traffic_light_status == "Red":
                    box_color = (0, 0, 255)
                elif detect_violation.traffic_light_status == "Green":
                    box_color = (0, 255, 0)
                else:
                    box_color = (0, 0, 0)
                cvzone.putTextRect(processed_frame, f"Traffic light status: {detect_violation.traffic_light_status}, L: {detect_violation.traffic_light_violator_counter}, W: {detect_violation.wrong_way_violator_counter}", (25, 60), scale=1, thickness=1, offset=3, colorR=box_color)
        
        if violation_type == "helmet":
            processed_frame = detect_violation.start_detect(frame)
            cvzone.putTextRect(processed_frame, f"Violation Counter: {detect_violation.helmet_violation_counter}", (25, 60), scale=1, thickness=1, offset=3)
        
        out.write(processed_frame)
    
    capture.release()
    out.release()
    
    return output_file_path

def draw_detected_areas(frame, areas):
    if areas:
        for area in areas:
            coords = area["coords"]
            north_count = area["north_count"]
            south_count = area["south_count"]
            status_dir = area["status_dir"]
            
            if len(coords) > 1:
                for i in range(len(coords) - 1):
                    cv2.line(frame, (coords[i][0], coords[i][1]), (coords[i + 1][0], coords[i + 1][1]), (255, 255, 255), 2)
            
            cvzone.putTextRect(frame, f"{north_count} {south_count} {status_dir}", (coords[0][0], coords[0][1] - 10), scale=0.8, thickness=1, offset=3)