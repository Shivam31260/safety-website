from flask import Flask, render_template, request, redirect, url_for, session, Response
import pathlib
import cv2
import numpy as np
from PIL import Image
import io
import torch
import os
import threading
from werkzeug.utils import secure_filename
import pygame  # Import pygame for MP3 playback
from datetime import datetime
import time

# Import model loaders
from helmet_head import load_model as load_helmet_head_model
from facemask import facemask_model as load_facemask_model
from firedetection import loadfiredetection_model as load_fire_detection_model

# Temporary fix for WindowsPath in YOLO
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
app.secret_key = 'Shivam31260'

# Dummy credentials for login
credentials = {
    "admin": "Shivam",
    "user1": "31260"
}

# Global variables
model = None
is_live_feed_active = False

# Set up the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Pygame mixer for MP3 playback
pygame.mixer.init()

def allowed_file(filename):
    """Check if the uploaded file is a valid image type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_selected_model(model_type):
    """Load the selected model based on the user's choice."""
    global model
    if model_type == 'safety':
        model = load_helmet_head_model()
    elif model_type == 'facemask':
        model = load_facemask_model()
    elif model_type == 'fire':
        model = load_fire_detection_model()
    else:
        raise ValueError("Invalid model type selected")

@app.route('/reset_model', methods=['POST'])
def reset_model():
    """Reset the model to None."""
    global model
    model = None
    print("Model has been reset.")  # Debugging
    return '', 204  # No content response

# Set up a directory for saving captured frames
CAPTURED_FRAMES_DIR = r'captured_frames'

def gen():
    """Generate live video feed with object detection."""
    global is_live_feed_active
    cap = cv2.VideoCapture(0)  # Capture from default camera (webcam)
    
    if not cap.isOpened():
        print("Error: Camera not found or failed to open.")
        return

    # Ensure the directory for captured frames exists
    os.makedirs(CAPTURED_FRAMES_DIR, exist_ok=True)

    while cap.isOpened() and is_live_feed_active:
        success, frame = cap.read()
        if success:
            # Convert frame to PIL Image for YOLO processing
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            img = Image.open(io.BytesIO(frame))
            results = model(img, size=640)  # Run the model on the image
            img = np.squeeze(results.render())  # Render the results

            # Check detected classes
            detected_classes = results.names
            detected_labels = results.pred[0][:, -1].tolist()  # Get detected class labels

            # Debugging log to see detected classes
            print(f"Detected labels: {detected_labels}")
            
            # Trigger sound and save frame if a certain class is detected
            for label in detected_labels:
                class_name = detected_classes[label]
                if class_name == 'head':  # Replace 'helmet' with your desired class
                    print(f"Class detected: {class_name}")

                    # Play the alert sound
                    # pygame.mixer.music.load(r'static\ALERT.mp3')  # Path to your MP3 file
                    pygame.mixer.music.load(os.path.join('static', 'ALERT.mp3'))
                    pygame.mixer.music.play()

                    # Save the current frame with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(CAPTURED_FRAMES_DIR, f"detected_{class_name}_{timestamp}.jpg")
                    img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
                    cv2.imwrite(save_path, img_BGR)  # Save the image
                    print(f"Captured and saved frame: {save_path}")

                elif class_name == 'without_mask':  # Replace 'helmet' with your desired class
                    print(f"Class detected: {class_name}")

                    # Play the alert sound
                    # pygame.mixer.music.load(r'static\ALERT.mp3')  # Path to your MP3 file
                    pygame.mixer.music.load(os.path.join('static', 'ALERT.mp3'))
                    pygame.mixer.music.play()

                    # Save the current frame with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(CAPTURED_FRAMES_DIR, f"detected_{class_name}_{timestamp}.jpg")
                    img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
                    cv2.imwrite(save_path, img_BGR)  # Save the image
                    print(f"Captured and saved frame: {save_path}")

                elif class_name == 'Fire':  # Replace 'helmet' with your desired class
                    print(f"Class detected: {class_name}")

                    # Play the alert sound
                    # pygame.mixer.music.load(r'static\ALERT.mp3')  # Path to your MP3 
                    pygame.mixer.music.load(os.path.join('static', 'ALERT.mp3'))
                    pygame.mixer.music.play()

                    # Save the current frame with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(CAPTURED_FRAMES_DIR, f"detected_{class_name}_{timestamp}.jpg")
                    img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
                    cv2.imwrite(save_path, img_BGR)  # Save the image
                    print(f"Captured and saved frame: {save_path}")


            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
            ret, buffer = cv2.imencode('.jpg', img_BGR)  # Encode the frame to JPEG
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            print("Error: Failed to read frame from camera.")
            break
    cap.release()

@app.route('/')
def login():
    """Display the login page."""
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def handle_login():
    """Handle login form submission."""
    username = request.form['username']
    password = request.form['password']
    if username in credentials and credentials[username] == password:
        session['username'] = username
        return redirect(url_for('selection'))
    else:
        return render_template('login.html', error="Invalid credentials. Please try again.")

@app.route('/selection', methods=['GET', 'POST'])
def selection():
    """Selection page to choose model for detection."""
    if request.method == 'POST':
        model_type = request.form['model']
        try:
            load_selected_model(model_type)  # Load the selected model
            return redirect(url_for('home'))
        except ValueError as e:
            return render_template('selection.html', error=str(e))
    return render_template('selection.html')

@app.route('/home')
def home():
    """Home page where detection options are displayed."""
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/toggle_feed', methods=['POST'])
def toggle_feed():
    """Start or stop live video feed."""
    global is_live_feed_active
    is_live_feed_active = not is_live_feed_active
    status = "started" if is_live_feed_active else "stopped"
    print(f"Live feed {status}")  # Debugging
    return '', 204

@app.route('/video')
def video():
    """Stream live video feed."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
def logout():
    """Logout the user and stop the live feed."""
    global is_live_feed_active
    is_live_feed_active = False
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return "No file part", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        img = Image.open(file_path)
        results = model(img, size=640)
        img = np.squeeze(results.render())
        img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', img_BGR)
        img_bytes = buffer.tobytes()
        
        return Response(img_bytes, mimetype='image/jpeg')
    
    return "Invalid file type", 400


if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000)
