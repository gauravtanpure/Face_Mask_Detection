# app.py - Complete Updated Version with Login System
from flask import Flask, render_template, request, jsonify, Response, send_from_directory, redirect, url_for, session, flash
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
import threading
import time
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure random key in production

# Static login credentials
ADMIN_USERNAME = "Admin"
ADMIN_PASSWORD = "123"

# Load the model and face detector
model = None
face_cascade = None

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Video streaming variables
global_frame = None
camera = None
stop_thread = False

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/static/sounds/<path:filename>')
def serve_sounds(filename):
    return send_from_directory('static/sounds', filename)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_mask(image_path):
    if model is None:
        return None
    
    try:
        test_image = load_img(image_path, target_size=(150, 150))
        test_image = img_to_array(test_image)
        test_image = test_image / 255.0  # Normalize
        test_image = np.expand_dims(test_image, axis=0)
        prediction = model.predict(test_image, verbose=0)[0][0]
        return float(prediction)
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None

def video_stream():
    global global_frame, camera, stop_thread
    
    camera = cv2.VideoCapture(0)
    while not stop_thread:
        ret, frame = camera.read()
        if not ret:
            break
            
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Process each face
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            
            # Save temporary image for prediction
            temp_path = 'temp_face.jpg'
            cv2.imwrite(temp_path, face_img)
            
            # Predict
            prediction = predict_mask(temp_path)
            if prediction is not None:
                has_mask = prediction < 0.5
                
                # Draw rectangle and label
                color = (0, 255, 0) if has_mask else (0, 0, 255)
                label = "MASK" if has_mask else "NO MASK"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Store the processed frame
        _, buffer = cv2.imencode('.jpg', frame)
        global_frame = buffer.tobytes()
        
        time.sleep(0.05)  # Control frame rate
    
    camera.release()

def generate_frames():
    global global_frame
    while True:
        if global_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n')
        time.sleep(0.05)

try:
    model = load_model('mymodel.keras')
    print("Model loaded successfully")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception as e:
    print(f"Error loading model or cascade: {e}")

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            session['username'] = ADMIN_USERNAME
            next_url = request.args.get('next')
            return redirect(next_url or url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
@login_required
def start_detection():
    global stop_thread
    stop_thread = False
    threading.Thread(target=video_stream).start()
    return jsonify({'status': 'started'})

@app.route('/stop_detection')
@login_required
def stop_detection():
    global stop_thread
    stop_thread = True
    return jsonify({'status': 'stopped'})

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        prediction = predict_mask(filepath)
        if prediction is None:
            return jsonify({'error': 'Error processing image'}), 500
        
        has_mask = prediction < 0.5  # Adjust threshold as needed
        confidence = (1 - prediction) if has_mask else prediction
        confidence = round(confidence * 100, 2)
        
        return jsonify({
            'has_mask': bool(has_mask),
            'confidence': confidence,
            'image_url': f'/static/uploads/{filename}'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/train', methods=['POST'])
@login_required
def train_model():
    return jsonify({
        'status': 'success',
        'message': 'Model training complete',
        'accuracy': 0.95  # Simulated accuracy
    })

if __name__ == '__main__':
    app.run(debug=True)