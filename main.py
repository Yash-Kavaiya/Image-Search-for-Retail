import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
import logging
import base64
from PIL import Image
import io

import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from werkzeug.utils import secure_filename
from gemini_service import ITSupportAgent

# Load environment variables
load_dotenv()

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Configure logging
# Example:
# @app.get("/hello")
# async def read_root():
#     return {"Hello": "World"}

@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})

@app.get('/my-trips', response_class=HTMLResponse)
async def my_trips(request: Request):
    return templates.TemplateResponse('my-trips.html', {"request": request})

@app.get('/travel-information', response_class=HTMLResponse)
async def travel_information(request: Request):
    return templates.TemplateResponse('travel-information.html', {"request": request})

@app.get('/destinations', response_class=HTMLResponse)
async def destinations(request: Request):
    return templates.TemplateResponse('destinations.html', {"request": request})

@app.get('/executive-club', response_class=HTMLResponse)
async def executive_club(request: Request):
    return templates.TemplateResponse('executive-club.html', {"request": request})
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

# You can add more FastAPI routes or configurations below if needed
# Example:
# @app.get("/hello")
# async def read_root():
#     return {"Hello": "World"}

@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})

@app.get('/my-trips', response_class=HTMLResponse)
async def my_trips(request: Request):
    return templates.TemplateResponse('my-trips.html', {"request": request})

@app.get('/travel-information', response_class=HTMLResponse)
async def travel_information(request: Request):
    return templates.TemplateResponse('travel-information.html', {"request": request})

@app.get('/destinations', response_class=HTMLResponse)
async def destinations(request: Request):
    return templates.TemplateResponse('destinations.html', {"request": request})

@app.get('/executive-club', response_class=HTMLResponse)
async def executive_club(request: Request):
    return templates.TemplateResponse('executive-club.html', {"request": request})

@app.post('/chat')
async def chat(request: Request):
    try:
        data = await request.json()
        if not data:
            raise HTTPException(status_code=400, detail="No data received")

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
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
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
        return {"response": response}

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request. Please try again.")

@app.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")
    
    if file and allowed_file(file.filename):
        # Create unique filename to prevent collisions
        original_filename = secure_filename(file.filename)
        filename = str(uuid.uuid4()) + '_' + original_filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        try:
            # Save file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Get file info
            file_size = os.path.getsize(file_path)
            file_type = 'image' if is_image_file(original_filename) else 'text' if is_text_file(original_filename) else 'other'
            
            logger.info(f"File uploaded successfully: {filename} (type: {file_type}, size: {file_size} bytes)")
            
            return {
                'success': True,
                'filename': filename,
                'original_filename': original_filename,
                'file_type': file_type,
                'url': f'/static/uploads/{filename}'
            }
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to save file")
    
    raise HTTPException(status_code=400, detail="File type not allowed")

@app.post('/upload-screenshot')
async def upload_screenshot(request: Request):
    try:
        data = await request.json()
        if not data or 'screenshot' not in data:
            raise HTTPException(status_code=400, detail="No screenshot data received")
        
        # Extract base64 image data
        screenshot_data = data['screenshot']
        
        # Remove data URL prefix if present
        if screenshot_data.startswith('data:image'):
            screenshot_data = screenshot_data.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(screenshot_data)
        
        # Create unique filename
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save the image
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        logger.info(f"Screenshot saved: {filename}")
        
        return {
            'success': True,
            'filename': filename,
            'url': f'/static/uploads/{filename}'
        }
        
    except Exception as e:
        logger.error(f"Error processing screenshot: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process screenshot")

@app.post('/cleanup')
async def cleanup_old_files():
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
        
        return {'success': True, 'message': 'Cleanup completed'}
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        raise HTTPException(status_code=500, detail="Cleanup failed")

@app.get('/health')
async def health_check():
    return {
        'status': 'healthy',
        'service': 'IT Support Agent',
        'version': '1.0.0'
    }

if __name__ == "__main__":
    # Use the PORT environment variable provided by Cloud Run, defaulting to 8080
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))