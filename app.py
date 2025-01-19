from flask import Flask, render_template, request, redirect, url_for, session, Response
import os
import cv2
import numpy as np
from PIL import Image
import io
import torch
import pygame  # Import pygame for MP3 playback
from datetime import datetime
import bcrypt  # For password hashing
import sqlite3
from werkzeug.utils import secure_filename

# Import model loaders
from helmet_head import load_model as load_helmet_head_model
from facemask import facemask_model as load_facemask_model
from firedetection import loadfiredetection_model as load_fire_detection_model

app = Flask(__name__)
app.secret_key = 'Shivam31260'

# Global variables
model = None
is_live_feed_active = False

# Set up folders and allowed extensions
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
CAPTURED_FRAMES_DIR = os.path.join(BASE_DIR, 'captured_frames')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CAPTURED_FRAMES_DIR, exist_ok=True)

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
    return '', 204  # No content response

def gen():
    """Generate live video feed with object detection."""
    global is_live_feed_active
    cap = cv2.VideoCapture(0)  # Capture from default camera

    if not cap.isOpened():
        print("Error: Camera not found or failed to open.")
        return

    while cap.isOpened() and is_live_feed_active:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame from camera.")
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        img = Image.open(io.BytesIO(frame))
        results = model(img, size=640)  # Run model on the image
        img = np.squeeze(results.render())

        detected_classes = results.names
        detected_labels = results.pred[0][:, -1].tolist()  # Get detected class labels

        for label in detected_labels:
            class_name = detected_classes[label]
            if class_name in ['head', 'without_mask', 'Fire']:
                pygame.mixer.music.load(os.path.join(STATIC_FOLDER, 'ALERT.mp3'))
                pygame.mixer.music.play()

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(CAPTURED_FRAMES_DIR, f"detected_{class_name}_{timestamp}.jpg")
                img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, img_BGR)
                print(f"Captured and saved frame: {save_path}")

        img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', img_BGR)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def login():
    """Display the login page."""
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def handle_login():
    """Handle login form submission."""
    username = request.form['username']
    password = request.form['password'].encode('utf-8')

    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()

    if row and bcrypt.checkpw(password, row[0]):
        session['username'] = username
        return redirect(url_for('selection'))
    else:
        return render_template('login.html', error="Invalid credentials. Please try again.")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Sign-Up Page"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())

        try:
            with sqlite3.connect("users.db") as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
                conn.commit()
                return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('signup.html', error="Username already exists. Please choose another.")

    return render_template('signup.html')

@app.route('/selection', methods=['GET', 'POST'])
def selection():
    """Selection page to choose model for detection."""
    if request.method == 'POST':
        model_type = request.form['model']
        try:
            load_selected_model(model_type)
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
    """Detect objects in an uploaded image."""
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
    app.run(debug=False, host="0.0.0.0")
