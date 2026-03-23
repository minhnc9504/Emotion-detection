"""
Emotion Detection Deployment Server
Flask web server for real-time emotion detection from webcam and image upload.
"""
import os
import cv2
import numpy as np
import base64
import io
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# Emotion labels matching the training data
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_EMOJIS = {
    'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨',
    'Happy': '😊', 'Sad': '😢', 'Surprise': '😲', 'Neutral': '😐'
}
EMOTION_COLORS = {
    'Angry': '#e74c3c', 'Disgust': '#27ae60', 'Fear': '#9b59b6',
    'Happy': '#f1c40f', 'Sad': '#3498db', 'Surprise': '#e67e22', 'Neutral': '#95a5a6'
}

# Build the CNN model (same architecture as emotions.py training)
def build_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    return model

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'src', 'model.h5')
cascade_path = os.path.join(os.path.dirname(__file__), 'src', 'haarcascade_frontalface_default.xml')

model = None
face_detector = None

try:
    if os.path.exists(MODEL_PATH):
        from tensorflow import keras
        model = keras.models.load_model(MODEL_PATH)
        print(f"[OK] Loaded trained model from {MODEL_PATH}")
    else:
        model = build_model()
        print("[INFO] No trained model found. Using untrained model for demonstration.")
except Exception as e:
    print(f"[WARN] Could not load model: {e}")
    model = build_model()

try:
    face_detector = cv2.CascadeClassifier(cascade_path)
    print("[OK] Loaded Haar Cascade classifier")
except Exception as e:
    print(f"[WARN] Could not load face detector: {e}")
    face_detector = None


def preprocess_face(face_img, target_size=(48, 48)):
    """Preprocess a face image for the model."""
    if face_img is None or face_img.size == 0:
        return None
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    normalized = resized / 255.0
    return normalized.reshape(1, 48, 48, 1)


def detect_and_predict(frame):
    """Detect faces and predict emotions."""
    if face_detector is None:
        return None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None, None

    results = []
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        processed = preprocess_face(face_roi)

        if processed is not None and model is not None:
            prediction = model.predict(processed, verbose=0)[0]
            emotion_idx = np.argmax(prediction)
            confidence = float(prediction[emotion_idx])
            emotion = EMOTION_LABELS[emotion_idx]
            results.append({
                'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                'emotion': emotion,
                'confidence': confidence,
                'emoji': EMOTION_EMOJIS[emotion],
                'color': EMOTION_COLORS[emotion],
                'all_probs': {EMOTION_LABELS[i]: float(prediction[i]) for i in range(7)}
            })

    return faces, results


def predict_from_image(image_data):
    """Predict emotion from a base64 encoded image."""
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        pil_img = Image.open(io.BytesIO(img_bytes))
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        faces, results = detect_and_predict(frame)
        return results if results else None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect():
    """API endpoint for emotion detection from image data."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    results = predict_from_image(data['image'])
    if results:
        return jsonify({'success': True, 'results': results})
    return jsonify({'success': False, 'message': 'No faces detected'}), 200


@app.route('/api/upload', methods=['POST'])
def upload():
    """API endpoint for file upload."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        faces, results = detect_and_predict(frame)

        if results:
            # Draw results on image
            for r in results:
                x, y, w, h = r['x'], r['y'], r['w'], r['h']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{r['emoji']} {r['emotion']} ({r['confidence']*100:.1f}%)"
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Encode result image
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({'success': True, 'results': results, 'image': f'data:image/jpeg;base64,{img_base64}'})
        return jsonify({'success': False, 'message': 'No faces detected'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'running', 'model_loaded': model is not None})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Emotion Detection Deployment Server")
    print("="*50)
    print(f"  Model: {'Loaded' if os.path.exists(MODEL_PATH) else 'Untrained (demo)'}")
    print(f"  Face Detector: {'OK' if face_detector else 'Not loaded'}")
    print("="*50)
    print("  Server running at: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
