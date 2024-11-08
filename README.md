
---

# DetectObject Recognition System
This project is an object recognition and detection system designed to recognize and detect objects in images, videos, or live webcam feed. The system allows for training a model with custom data, uploading and detecting objects in media files, and performing real-time detection using a webcam..

## Table of Contents
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [API Overview](#api-overview)
- [License](#license)

## Features

- **Image, Video, and Webcam Detection**: Detect the Object in images, videos, or live webcam feed.
- **Custom Model Training**: Upload images and annotate for custom model training.
- **Interactive Interface**: Simple, user-friendly HTML interface.

## Setup

1. **Clone this Repository**:
   ```bash
   git clone <repository-url>
   cd object-recognition-system
   ```

2. **Create Virtual Environment (Optional)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `uploads` Directory**:
   ```bash
   mkdir uploads
   ```

5. **Start the Server**:
   ```bash
   python app.py
   ```

6. **Access the Application**:
   Open `http://127.0.0.1:5000` in a web browser.

## Usage

### 1. Home Page
- **Train**: Start training a custom model.
- **Detect**: Detect PPE in uploaded images, videos, or webcam feed.

### 2. Training Page
- Upload multiple images.
- Click **Annotate Images** to mark PPE items.
- Click **Train Model** to train on uploaded images.

### 3. Detection Page
- Choose to detect from:
  - **Image**: Upload an image and click **Detect from Image**.
  - **Video**: Upload a video file and click **Detect from Video**.
  - **Webcam**: Use the webcam for real-time PPE detection.
- **Results Table**: Shows detected PPE classes and confidence levels.

## API Overview

| Endpoint               | Method | Description                            |
|------------------------|--------|----------------------------------------|
| `/train`               | GET    | Open training page                     |
| `/detect`              | GET    | Open detection page                    |
| `/annotate_images`     | POST   | Upload and annotate images for training|
| `/train_model`         | POST   | Train the model                        |
| `/detect_image`        | POST   | Detect objects in an image             |
| `/detect_video`        | POST   | Detect objects in a video              |
| `/detect_webcam`       | GET    | Real-time detection with webcam        |

## License

This project is licensed under the MIT License.

---
