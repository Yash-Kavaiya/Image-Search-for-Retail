from flask import Flask, render_template, request, jsonify, url_for, redirect
import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
import logging
from werkzeug.utils import secure_filename
from gemini_service import ITSupportAgent
import base64
from PIL import Image
import io

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = 'application/json'
app.config['JSON_SORT_KEYS'] = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize IT Support Agent
it_support_agent = ITSupportAgent()

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'xls', 'xlsx', 'log', 'csv'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
ALLOWED_TEXT_EXTENSIONS = {'txt', 'log', 'csv', 'conf', 'cfg', 'ini'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def is_text_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_TEXT_EXTENSIONS

def get_mime_type(filename):
    """Get MIME type based on file extension"""
    ext = filename.rsplit('.', 1)[1].lower()
    mime_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp'
    }
    return mime_types.get(ext, 'application/octet-stream')

def read_text_file(filepath):
    """Read content from text file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read()
        except:
            return "Error reading file content"

@app.route('/')
def home():
    return render_template('index.html')

# Additional routes
@app.route('/my-trips')
def my_trips():
    return render_template('my-trips.html')

@app.route('/travel-information')
def travel_information():
    return render_template('travel-information.html')

@app.route('/destinations')
def destinations():
    return render_template('destinations.html')

@app.route('/executive-club')
def executive_club():
    return render_template('executive-club.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data received"}), 400

        message = data.get('message', '')
        attachment = data.get('attachment', None)
        
        # Initialize variables for processing
        image_data = None
        image_mime_type = None
        file_content = None
        
        # Process attachment if present
        if attachment:
            # Handle screenshot or uploaded file
            if attachment.get('isScreenshot'):
                # For screenshots, the data is usually sent as base64
                # This would need to be implemented based on how your frontend sends screenshot data
                logger.info("Processing screenshot attachment")
            else:
                # For regular file uploads, we need to check if file exists
                # This assumes the file was already uploaded via /upload endpoint
                filename = attachment.get('filename')
                if filename:
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    if os.path.exists(filepath):
                        if is_image_file(filename):
                            # Read image file
                            with open(filepath, 'rb') as f:
                                image_data = f.read()
                            image_mime_type = get_mime_type(filename)
                            logger.info(f"Processing image file: {filename}")
                        elif is_text_file(filename):
                            # Read text file
                            file_content = read_text_file(filepath)
                            logger.info(f"Processing text file: {filename}")
        
        # Log the incoming request for debugging
        logger.info(f"Received message: {message[:100]}..." if len(message) > 100 else f"Received message: {message}")
        if image_data:
            logger.info(f"Image attached: {image_mime_type}")
        if file_content:
            logger.info(f"File content length: {len(file_content)} characters")
        
        # Get response from IT Support Agent
        try:
            response = it_support_agent.get_support_response_sync(
                text_query=message,
                image_data=image_data,
                image_mime_type=image_mime_type,
                file_content=file_content
            )
        except Exception as gemini_error:
            logger.error(f"Gemini API error: {str(gemini_error)}")
            response = "I apologize, but I'm having trouble connecting to the support service. Please check your internet connection and try again."
        
        # Return the response
        return jsonify({"response": response})

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request. Please try again."}), 500

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Create unique filename to prevent collisions
        original_filename = secure_filename(file.filename)
        filename = str(uuid.uuid4()) + '_' + original_filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(file_path)
            
            # Get file info
            file_size = os.path.getsize(file_path)
            file_type = 'image' if is_image_file(original_filename) else 'text' if is_text_file(original_filename) else 'other'
            
            logger.info(f"File uploaded successfully: {filename} (type: {file_type}, size: {file_size} bytes)")
            
            return jsonify({
                'success': True,
                'filename': filename,
                'original_filename': original_filename,
                'file_type': file_type,
                'url': url_for('static', filename=f'uploads/{filename}')
            })
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return jsonify({'error': 'Failed to save file'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

# Route to handle screenshot uploads
@app.route('/upload-screenshot', methods=['POST'])
def upload_screenshot():
    try:
        data = request.json
        if not data or 'screenshot' not in data:
            return jsonify({'error': 'No screenshot data received'}), 400
        
        # Extract base64 image data
        screenshot_data = data['screenshot']
        
        # Remove data URL prefix if present
        if screenshot_data.startswith('data:image'):
            screenshot_data = screenshot_data.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(screenshot_data)
        
        # Create unique filename
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the image
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        logger.info(f"Screenshot saved: {filename}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'url': url_for('static', filename=f'uploads/{filename}')
        })
        
    except Exception as e:
        logger.error(f"Error processing screenshot: {str(e)}")
        return jsonify({'error': 'Failed to process screenshot'}), 500

# Clean up old files periodically (optional)
@app.route('/cleanup', methods=['POST'])
def cleanup_old_files():
    """Clean up files older than 24 hours"""
    try:
        import time
        current_time = time.time()
        
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > 86400:  # 24 hours
                    os.remove(file_path)
                    logger.info(f"Deleted old file: {filename}")
        
        return jsonify({'success': True, 'message': 'Cleanup completed'})
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        return jsonify({'error': 'Cleanup failed'}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'IT Support Agent',
        'version': '1.0.0'
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))