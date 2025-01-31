from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import cv2
import numpy as np
import threading
from werkzeug.utils import secure_filename
from preprocess import remove_noise, deblur_image, upscale_image

# Initialize Flask app
app = Flask(__name__)

# Define folders
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/restored'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE_MB = 10  # Maximum file size (in MB)

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Thread lock for concurrent processing
lock = threading.Lock()

def allowed_file(filename):
    """Check if the uploaded file is an allowed image type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main upload page."""
    return render_template('index.html')

@app.route('/restore', methods=['POST'])
def restore_image():
    """Handles image restoration by applying noise removal, deblurring, and super-resolution."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):             
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG'}), 400
    
    if len(file.read()) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return jsonify({'error': f'File too large. Max size is {MAX_FILE_SIZE_MB}MB'}), 400
    file.seek(0)  # Reset file read pointer

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Read image
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400

    # Thread-safe processing
    with lock:
        try:
            img = remove_noise(img)
            img = deblur_image(img)
            img = upscale_image(img)
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    # Save restored image
    restored_filename = f'restored_{filename}'
    restored_path = os.path.join(PROCESSED_FOLDER, restored_filename)
    cv2.imwrite(restored_path, img)

    return jsonify({'original': filename, 'restored': restored_filename})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve processed images."""
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 4000)), debug=True, threaded=True)

