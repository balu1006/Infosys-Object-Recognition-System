from ultralytics import YOLO
from flask import Flask, request, Response, jsonify
from waitress import serve
from PIL import Image
import cv2
import numpy as np
import json
import io
import base64
import tempfile
import os

from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
import re
from time import time

app = Flask(__name__)

app.secret_key = 'Object-recognition-System' 
app.config["MONGO_URI"] = "mongodb://localhost:27017/ORS"
mongo = PyMongo(app)

# Initialize the model
model = YOLO("best - Copy.pt") 

# Directory to save annotated images temporarily
ANNOTATED_DIR = 'annotated_images'
if not os.path.exists(ANNOTATED_DIR):
    os.makedirs(ANNOTATED_DIR)

# Directory to save the training dataset
TRAINING_DATASET_DIR = 'training_dataset'
if not os.path.exists(TRAINING_DATASET_DIR):
    os.makedirs(TRAINING_DATASET_DIR)

# @app.route("/")
# def root():
#     try:
#         with open("first.html") as file:
#             return file.read()
#     except FileNotFoundError:
#         return Response(
#             json.dumps({"error": "first.html not found"}), 
#             mimetype='application/json'
#         ), 404


# Route for Landing page (home page with options)
@app.route('/')
def root():
    try:
        with open("landing.html") as file:
            return file.read()
    except FileNotFoundError:
        return Response(
            json.dumps({"error": "landing.html not found"}), 
            mimetype='application/json'
        ), 404
        
@app.route("/about")
def about():
    try:
        with open("AboutUS.html") as file:
            return file.read()
    except FileNotFoundError:
        return Response(
            json.dumps({"error": "AboutUs.html not found"}), 
            mimetype='application/json'
        ), 404
        
@app.route("/contact")
def contact():
    try:
        with open("Contact.html") as file:
            return file.read()
    except FileNotFoundError:
        return Response(
            json.dumps({"error": "Contact.html not found"}), 
            mimetype='application/json'
        ), 404

# Route for Sign Up page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('signup'))

        # Check if the username or email already exists in MongoDB
        existing_user = mongo.db.users.find_one({'$or': [{'username': username}, {'email': email}]})
        if existing_user:
            flash('Username or Email already exists!', 'danger')
            return redirect(url_for('signup'))

        # Hash the password before saving
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Create a new user document and insert it into the MongoDB 'users' collection
        new_user = {'username': username, 'email': email, 'password': hashed_password}
        mongo.db.users.insert_one(new_user)

        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')


# Route for Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the user exists in MongoDB
        user = mongo.db.users.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['username'] = username  # Store the username in the session
            flash('Login successful!', 'success')
            return redirect(url_for('first'))  # Redirect to the first.html page after login
        else:
            flash('Invalid username or password!', 'danger')

    return render_template('login.html')

@app.route('/first')
def first():
    try:
        with open("first.html") as file:
            return file.read()
    except FileNotFoundError:
        return Response(
            json.dumps({"error": "first.html not found"}), 
            mimetype='application/json'
        ), 404

@app.route("/train")
def train():
    try:
        with open("train.html") as file:
            return file.read()
    except FileNotFoundError:
        return Response(
            json.dumps({"error": "train.html not found"}), 
            mimetype='application/json'
        ), 404

@app.route("/detect")
def detect():
    try:
        with open("index.html") as file:
            return file.read()
    except FileNotFoundError:
        return Response(
            json.dumps({"error": "index.html not found"}), 
            mimetype='application/json'
        ), 404

@app.route("/annotate_images", methods=["POST"])
def annotate_images():
    if "images" not in request.files:
        return jsonify({"error": "No images provided"}), 400

    try:
        files = request.files.getlist("images")
        annotated_files = []

        for file in files:
            image = Image.open(file.stream).convert("RGB")
            boxes, img_str = detect_objects_on_image(image)

            # Save the annotated image to a file
            annotated_image = Image.open(io.BytesIO(base64.b64decode(img_str)))
            annotated_path = os.path.join(ANNOTATED_DIR, file.filename)
            annotated_image.save(annotated_path)
            annotated_files.append(annotated_path)

            # Create a corresponding annotation file
            annotation_path = os.path.splitext(annotated_path)[0] + '.txt'
            with open(annotation_path, 'w') as ann_file:
                for box in boxes:
                    x1, y1, x2, y2, class_name, prob = box
                    # Normalize coordinates for YOLO format
                    width = x2 - x1
                    height = y2 - y1
                    x_center = (x1 + x2) / 2 / image.width
                    y_center = (y1 + y2) / 2 / image.height
                    norm_width = width / image.width
                    norm_height = height / image.height
                    
                    # Write class (assuming class names are mapped to integers)
                    class_id = get_class_id(class_name)  # Implement this function to map class names to IDs
                    ann_file.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")

        return jsonify({"message": "Annotation completed successfully", "annotated_files": annotated_files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train_model", methods=["POST"])
def train_model():
    try:
        # Start the training process using the annotated images
        model.train(data=prepare_training_data(), epochs=5)  # Adjust epochs as needed

        return jsonify({"message": "Training completed successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/detect_image", methods=["POST"])
def detect_image():
    if "image_file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        image_file = request.files["image_file"]
        image = Image.open(image_file.stream).convert("RGB")

        boxes, img_str = detect_objects_on_image(image)

        return jsonify({"boxes": boxes, "image": img_str})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def detect_objects_on_image(image):
    results = model.predict(image)
    result = results[0]
    output = []

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        class_name = result.names[class_id]

        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_cv, f"{class_name} {prob}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        output.append([x1, y1, x2, y2, class_name, prob])

    processed_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    buffered = io.BytesIO()
    processed_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return output, img_str

def get_class_id(class_name):
    class_mapping = {
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
        "5": 4,
        "6": 5,
        "7": 6,
        "Glass": 7,
        "Glasses": 8,
        "Gloves": 9,
        "Goggles": 10,
        "Helmet": 11,
        "Mask": 12,
        "No-Helmet": 13,
        "No-Vest": 14,
        "Person": 15,
        "Safety-Boot": 16,
        "Safety-Vest": 17,
        "Safety_shoes": 18,
        "Shoes": 19,
        "Vest": 20,
        "blue": 21,
        "combination": 22,
        "face_mask": 23,
        "face_nomask": 24,
        "glass": 25,
        "glasses": 26,
        "glove": 27,
        "hand_glove": 28,
        "hand_noglove": 29,
        "head": 30,
        "head_helmet": 31,
        "head_nohelmet": 32,
        "helmet": 33,
        "no helmet": 34,
        "no vest": 35,
        "no-vest": 36,
        "no_helmet": 37,
        "no_vest": 38,
        "person": 39,
        "protective_suit": 40,
        "red": 41,
        "safety boot": 42,
        "shoes": 43,
        "vest": 44,
        "white": 45,
        "worker": 46,
        "yellow": 47
    }

    if class_name in class_mapping:
        return class_mapping[class_name]
    else:
        print(f"Warning: '{class_name}' is not a recognized class name.")
        return None  

def prepare_training_data():
    class_mapping = {
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
        "5": 4,
        "6": 5,
        "7": 6,
        "Glass": 7,
        "Glasses": 8,
        "Gloves": 9,
        "Goggles": 10,
        "Helmet": 11,
        "Mask": 12,
        "No-Helmet": 13,
        "No-Vest": 14,
        "Person": 15,
        "Safety-Boot": 16,
        "Safety-Vest": 17,
        "Safety_shoes": 18,
        "Shoes": 19,
        "Vest": 20,
        "blue": 21,
        "combination": 22,
        "face_mask": 23,
        "face_nomask": 24,
        "glass": 25,
        "glasses": 26,
        "glove": 27,
        "hand_glove": 28,
        "hand_noglove": 29,
        "head": 30,
        "head_helmet": 31,
        "head_nohelmet": 32,
        "helmet": 33,
        "no helmet": 34,
        "no vest": 35,
        "no-vest": 36,
        "no_helmet": 37,
        "no_vest": 38,
        "person": 39,
        "protective_suit": 40,
        "red": 41,
        "safety boot": 42,
        "shoes": 43,
        "vest": 44,
        "white": 45,
        "worker": 46,
        "yellow": 47
    }

    
    class_names = list(class_mapping.keys())  
    yaml_content = f"""
    train: {os.path.abspath(ANNOTATED_DIR)}
    val: {os.path.abspath(ANNOTATED_DIR)}
    nc: {len(class_names)}  # Number of classes
    names: {json.dumps(class_names)}  # List of class names
    """

    yaml_file_path = os.path.join(TRAINING_DATASET_DIR, "dataset.yaml")
    with open(yaml_file_path, 'w') as yaml_file:
        yaml_file.write(yaml_content)

    return yaml_file_path


@app.route("/detect_video", methods=["POST"])
def detect_video():
    if "video_file" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    try:
        video_file = request.files["video_file"]
        return process_video(video_file)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/detect_webcam")
def detect_webcam():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/detect_webcam_results")
def detect_webcam_results():
    global detected_boxes
    return jsonify({"boxes": detected_boxes})

def gen_frames():
    cap = cv2.VideoCapture(0) 
    global detected_boxes  

    while True:
        success, frame = cap.read()  
        if not success:
            break

        # Detect objects on the current frame
        boxes, results = detect_objects_on_frame(frame)

        detected_boxes = boxes  

        for box in boxes:
            x1, y1, x2, y2, class_name, prob = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {prob}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release() 

def process_video(video_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_file.save(temp_file.name)

    cap = cv2.VideoCapture(temp_file.name)
    frames_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes, results = detect_objects_on_frame(frame)
        frames_data.append({"boxes": boxes, "results": results})

    cap.release()
    os.remove(temp_file.name)  # Clean up the temporary file
    return jsonify({"frames": frames_data})

def detect_objects_on_frame(frame):
    results = model.predict(frame)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        class_name = result.names[class_id]
        output.append([x1, y1, x2, y2, class_name, prob])
    return output, results

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)
