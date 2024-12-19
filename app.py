from flask import Flask, request, jsonify
from .utils import FileUtils
from .controllers import upload_video_controller, detect_line_violation_controller, detect_helmet_violation_controller, get_captured_violations_controller, get_file

app = Flask(__name__)

utils = FileUtils()
app.config['UPLOAD_FOLDER'] = utils.create_uploads_dir()

METHOD_NOT_ALLOWED_ERROR = 'Method Not Allowed'

@app.route("/")
def hello_world():
    return "Hello, World!"

@app.route('/upload', methods=['POST'])
def upload_video():
    if request.method == 'POST':
        return upload_video_controller(app.config['UPLOAD_FOLDER'], utils)
    else:
        return jsonify({'error': METHOD_NOT_ALLOWED_ERROR}), 405

@app.route('/detectLineViolation', methods=['POST'])
def detect_line_violation():
    if request.method == 'POST': 
        return detect_line_violation_controller(app.config['UPLOAD_FOLDER'], utils)
    else: return jsonify({'error': METHOD_NOT_ALLOWED_ERROR}), 405

@app.route('/detectHelmetViolation', methods=['POST'])
def detect_helmet_violation():
    if request.method == 'POST': 
        return detect_helmet_violation_controller(app.config['UPLOAD_FOLDER'], utils)
    else: return jsonify({'error': METHOD_NOT_ALLOWED_ERROR}), 405

@app.route('/capturedViolation', methods=['GET'])
def get_captured_violations():
    if request.method == 'GET':
        return get_captured_violations_controller(app.config['UPLOAD_FOLDER'], utils)
    else:
        return jsonify({'error': METHOD_NOT_ALLOWED_ERROR}), 405

@app.route('/file', methods=['GET'])
def get_video():
    if request.method == 'GET':
        return get_file(app.config['UPLOAD_FOLDER'])
    else:
        return jsonify({'error': METHOD_NOT_ALLOWED_ERROR}), 405