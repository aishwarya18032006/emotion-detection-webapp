from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import cv2
import os
import socket
from datetime import datetime

# ----------------------------
# Flask App Configuration
# ----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
os.makedirs(INSTANCE_DIR, exist_ok=True)

app = Flask(__name__,
            static_folder=os.path.join(BASE_DIR, "static"),
            template_folder=os.path.join(BASE_DIR, "templates"))

# ----------------------------
# Database Setup
# ----------------------------
db_path = os.path.join(INSTANCE_DIR, "emotions.db")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ----------------------------
# Database Model
# ----------------------------
class EmotionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    emotion = db.Column(db.String(50), nullable=False)
    image_path = db.Column(db.String(200), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_cnn.h5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def index():
    records = EmotionRecord.query.order_by(EmotionRecord.id.desc()).all()
    return render_template("index.html", records=records)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles emotion prediction from uploaded image"""
    if 'file' not in request.files:
        return render_template("index.html", message="⚠️ No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", message="⚠️ No file selected")

    # Save uploaded file
    upload_dir = os.path.join(app.static_folder, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    try:
        # Preprocess image
        img = cv2.imread(file_path)
        img = cv2.resize(img, (48, 48))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=(0, -1)) / 255.0

        prediction = model.predict(img)
        emotion = EMOTIONS[np.argmax(prediction)]

        # Save record
        record = EmotionRecord(emotion=emotion, image_path=f"uploads/{file.filename}")
        db.session.add(record)
        db.session.commit()

        records = EmotionRecord.query.order_by(EmotionRecord.id.desc()).all()
        return render_template(
            "index.html",
            prediction=emotion,
            image_path=f"uploads/{file.filename}",
            records=records
        )
    except Exception as e:
        return render_template("index.html", message=f"❌ Error processing image: {str(e)}")

@app.route('/webcam')
def webcam():
    """Run webcam detection (disabled on Render)"""
    hostname = socket.gethostname()

    # Detect if running on Render (no webcam support)
    if "render" in hostname.lower():
        return render_template("index.html",
                               message="⚠️ Webcam is not supported on Render. Please run locally in VS Code.")

    # Local webcam functionality
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return render_template("index.html", message="❌ Webcam not accessible. Check your camera permissions.")

    print("🎥 Press 'q' to quit webcam window.")

    webcam_dir = os.path.join(app.static_folder, "webcam_snaps")
    os.makedirs(webcam_dir, exist_ok=True)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = np.expand_dims(roi_gray, axis=(0, -1)) / 255.0

            prediction = model.predict(roi_gray)
            emotion = EMOTIONS[np.argmax(prediction)]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            if frame_count % 10 == 0:
                filename = f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                save_path = os.path.join(webcam_dir, filename)
                cv2.imwrite(save_path, frame)

                record = EmotionRecord(emotion=emotion, image_path=f"webcam_snaps/{filename}")
                db.session.add(record)
                db.session.commit()

        frame_count += 1
        cv2.imshow("Live Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    records = EmotionRecord.query.order_by(EmotionRecord.id.desc()).all()
    return render_template("index.html", records=records)

@app.route('/delete_history', methods=['POST'])
def delete_history():
    """Deletes all stored emotion records"""
    EmotionRecord.query.delete()
    db.session.commit()
    return render_template("index.html", records=[], message="🗑️ History cleared successfully!")

# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
