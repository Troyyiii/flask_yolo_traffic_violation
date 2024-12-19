import os
from flask import request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from detect import start_detection

NO_ID_ERROR = {
    'status': 'error',
    'message': 'No id part',
    'error_code': 400
}

def upload_video_controller(app_config, utils):
    if 'file' not in request.files:
        response = {
            'status': 'error',
            'message': 'No file part',
            'error_code': 400
        }
        return jsonify(response), 400
    
    if len(request.files.getlist('file')) > 1:
        response = {
            'status': 'error',
            'message': 'Only one file is allowed',
            'error_code': 400
        }
        return jsonify(response), 400
    
    file = request.files['file']
    if file.filename == '':
        response = {
            'status': 'error',
            'message': 'No selected file',
            'error_code': 400
        }
        return jsonify(response), 400
    
    if file and utils.allowed_file(file.filename):
        filename = secure_filename(file.filename)
        uid, unique_name, created_at = utils.upload_process(filename)
        folder_name = os.path.join(app_config, uid)
        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, f"temp_{unique_name}")
        file.save(file_path)
        response = {
            'status': 'success',
            'message': 'File uploaded successfully',
            'data': {
                'id': uid,
                'filename': unique_name,
                'file_path': file_path.replace('\\', '/').replace(app_config, ''),
                'created_at': created_at
            }
        }
        return jsonify(response), 201
    
    response = {
        'status': 'error',
        'message': 'Invalid file type',
        'error_code': 400
    }
    return jsonify(response), 400

def detect_line_violation_controller(app_config, utils):
    data = request.form
    
    if 'id' not in data or data['id'] == '':
        return jsonify(NO_ID_ERROR), 400
    
    idx = data['id']
    file_dir = utils.search_video_dir(app_config, idx)
    video_input_path = utils.search_video(app_config, idx)
    
    if file_dir and video_input_path:
        output_file_path = start_detection(file_dir, video_input_path, violation_type="line")
        filename = os.path.basename(output_file_path)
        response = {
            'status': 'success',
            'message': 'Line violation detection success',
            'data': {
                'id': idx,
                'filename': filename,
                'output_file_path': output_file_path.replace('\\', '/').replace(app_config, '')
            }
        }
        return jsonify(response), 201
    else:
        response = {
            'status': 'error',
            'message': 'Video not found',
            'error_code': 404
        }
        return jsonify(response), 404

def detect_helmet_violation_controller(app_config, utils):
    data = request.form
    
    if 'id' not in data or data['id'] == '':
        return jsonify(NO_ID_ERROR), 400
    
    idx = data['id']
    file_dir = utils.search_video_dir(app_config, idx)
    video_input_path = utils.search_video(app_config, idx)
    
    if file_dir and video_input_path:
        output_file_path = start_detection(file_dir, video_input_path, violation_type="helmet")
        filename = os.path.basename(output_file_path)
        response = {
            'status': 'success',
            'message': 'Helmet violation detection success',
            'data': {
                'id': idx,
                'filename': filename,
                'output_file_path': output_file_path.replace('\\', '/').replace(app_config, '')
            }
        }
        return jsonify(response), 201
    else:
        response = {
            'status': 'error',
            'message': 'Video not found',
            'error_code': 404
        }
        return jsonify(response), 404

def get_captured_violations_controller(app_config, utils):
    idx = request.args.get('id')
    
    if not idx:
        return jsonify(NO_ID_ERROR), 400
    
    result = utils.get_captured_violations(app_config, idx)
    
    if result:
        response = {
            'status': 'success',
            'message': 'Get captured violations success',
            'data': {
                'id': idx,
                'violations': result
            }
        }
        return jsonify(response), 201
    
    response = {
        'status': 'error',
        'message': 'No violations found for the provided id',
        'error_code': 404
    }
    return jsonify(response), 404

def get_file(app_config):
    filename = request.args.get('file')
    
    if not filename:
        response = {
            'status': 'error',
            'message': 'No file name provided',
            'error_code': 400
        }
        return jsonify(response), 400
    try:
        return send_from_directory(app_config, filename)
    except FileNotFoundError:
        response = {
            'status': 'error',
            'message': 'File not found',
            'error_code': 404
        }
        return jsonify(response), 404