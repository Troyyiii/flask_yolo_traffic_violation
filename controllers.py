import os
from flask import request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from .detect import start_detection

NO_ID_ERROR = 'No id part'

def upload_video_controller(app_config, utils):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    if len(request.files) > 1:
        return jsonify({'error': 'Only one file is allowed'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and utils.allowed_file(file.filename):
        filename = secure_filename(file.filename)
        uid, unique_name, created_at = utils.upload_process(filename)
        folder_name = os.path.join(app_config, uid)
        os.makedirs(folder_name, exist_ok=True)
        temp_file_path = os.path.join(folder_name, f"temp_{unique_name}")
        file.save(temp_file_path)
        file_path = os.path.join(folder_name, unique_name)
        utils.save_and_resize(temp_file_path, file_path)
        os.remove(temp_file_path)
        response = {
            'id': uid,
            'filename': unique_name,
            'file_path': file_path.replace('\\', '/').replace(app_config, ''),
            'created_at': created_at
        }
        return jsonify(response), 201
    return jsonify({'error': 'Invalid file type'}), 400

def detect_line_violation_controller(app_config, utils):
    data = request.form
    if 'id' not in data or data['id'] == '':
        return jsonify({'error': NO_ID_ERROR}), 400
    idx = data['id']
    file_dir = utils.search_video_dir(app_config, idx)
    video_input_path = utils.search_video(app_config, idx)
    if file_dir and video_input_path:
        output_file_path = start_detection(file_dir, video_input_path, violation_type="line")
        response = {
            'id': idx,
            'output_file_path': output_file_path.replace('\\', '/').replace(app_config, '')
        }
        return jsonify(response), 201
    else:
        return jsonify({'error': 'Video not found'}), 404

def detect_helmet_violation_controller(app_config, utils):
    data = request.form
    if 'id' not in data or data['id'] == '':
        return jsonify({'error': NO_ID_ERROR}), 400
    idx = data['id']
    file_dir = utils.search_video_dir(app_config, idx)
    video_input_path = utils.search_video(app_config, idx)
    if file_dir and video_input_path:
        output_file_path = start_detection(file_dir, video_input_path, violation_type="helmet")
        response = {
            'id': idx,
            'output_file_path': output_file_path.replace('\\', '/').replace(app_config, '')
        }
        return jsonify(response), 201
    else:
        return jsonify({'error': 'Video not found'}), 404

def get_captured_violations_controller(app_config, utils):
    idx = request.args.get('id')
    if not idx:
        return jsonify({'error': NO_ID_ERROR}), 400
    result = utils.get_captured_violations(app_config, idx)
    if not result:
        return jsonify({'error': 'No violations found for the provided id'}), 404
    response = {
        'id': idx,
        'violations': result
    }
    return jsonify(response), 201

def get_file(app_config):
    filename = request.args.get('file')
    if not filename:
        return jsonify({'error': 'No file name provided'}), 400
    try:
        return send_from_directory(app_config, filename)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404