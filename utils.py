import os
import cv2
import uuid
from datetime import datetime

class FileUtils:
    def __init__(self):
        self.UPLOAD_FOLDER = './uploads'
        self.ALLOWED_EXTENSIONS = {'mp4'}
    
    def create_uploads_dir(self):
        uploads_dir = os.path.join(self.UPLOAD_FOLDER)
        os.makedirs(uploads_dir, exist_ok=True)
        return uploads_dir
    
    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS
    
    def upload_process(self, filename):
        name, _ = os.path.splitext(filename)
        unique_id = uuid.uuid4().hex
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        
        filename = f"{unique_id}_{name}_{timestamp}.mp4"
        return unique_id, filename, timestamp
    
    def save_and_resize(self, temp_file_path, file_path, new_fps=15):
        capture = cv2.VideoCapture(temp_file_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        original_fps = capture.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(file_path, fourcc, new_fps, (1280, 720))
        
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / original_fps
        new_frame_count = int(duration * new_fps)
        
        for i in range(new_frame_count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, i * original_fps / new_fps)
            ret, frame = capture.read()
            if not ret:
                break
            frame = cv2.resize(frame, (1280, 720))
            out.write(frame)
        
        capture.release()
        out.release()
    
    def search_video_dir(self, app_config, id):
        for root, dirs, files in os.walk(app_config):
            for dir_name in dirs:
                if dir_name == id:
                    return os.path.join(root, dir_name)
        return None
    
    def search_video(self, app_config, id):
        for root, dirs, files in os.walk(app_config):
            for file in files:
                if file.endswith('.mp4'):
                    file_id = file.split('_')[0]
                    if file_id == id:
                        return os.path.join(root, file)
        return None
    
    def get_captured_violations(self, app_config, id):
        violation_dir = self.search_video_dir(app_config, id)
        if not violation_dir:
            return None
        
        traffic_violation_dir = os.path.join(violation_dir, 'traffic_violation')
        if not os.path.exists(traffic_violation_dir):
            return None
        
        violations = {
            'helmet': [],
            'traffic_line': [],
            'wrong_way': []
        }
        
        for category in violations.keys():
            category_dir = os.path.join(traffic_violation_dir, category)
            if os.path.exists(category_dir):
                for file in os.listdir(category_dir):
                    if file.endswith('.jpg'):
                        file_path = os.path.join(category_dir, file)
                        violations[category].append({
                            'filename': file,
                            'file_path': file_path.replace('\\', '/').replace(app_config, '')
                        })
        
        return violations