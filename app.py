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

app = Flask(__name__)

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

@app.route("/")
def root():
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
        model.train(data=prepare_training_data(), epochs=50)  # Adjust epochs as needed

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
        "person": 0,
        "vest": 1,
        "no-vest": 2,
        "helmet": 3,
        "mask": 4
    }
    return class_mapping.get(class_name, 99)

def prepare_training_data():
    class_mapping = {
        "person": 0,
        "vest": 1,
        "no-vest": 2,
        "helmet": 3,
        "mask": 4
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
