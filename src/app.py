import os
from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import csv
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)

# Paths
known_faces_path = "../known_faces.pkl"
attendance_file = "../attendance.csv"

# Load database
df = pd.read_pickle(known_faces_path)
known_embeddings = df["Embedding"].tolist()
known_names = df["Name"].tolist()

# Cooldown
cooldown_period = 300
last_logged = defaultdict(lambda: datetime.min)

cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            result = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True)
            embedding = result[0]["embedding"]
            face_coords = result[0]["facial_area"]
            
            distances = [np.linalg.norm(embedding - known_emb) for known_emb in known_embeddings]
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            threshold = 0.6
            name = known_names[min_distance_idx] if min_distance < threshold else "Unknown"

            x, y, w, h = face_coords["x"], face_coords["y"], face_coords["w"], face_coords["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            current_time = datetime.now()
            if name != "Unknown" and (current_time - last_logged[name]).total_seconds() > cooldown_period:
                with open(attendance_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    if os.path.getsize(attendance_file) == 0:
                        writer.writerow(["Name", "Timestamp"])
                    writer.writerow([name, current_time.strftime("%Y-%m-%d %H:%M:%S")])
                last_logged[name] = current_time

        except:
            pass

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
   
@app.route('/')
def index():
    return render_template_string("""
    <html>
        <head>
            <title>Attendance System</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; background-color: #f0f0f0; }
                h1 { color: #333; }
                pre { background-color: #fff; padding: 10px; border: 1px solid #ccc; display: inline-block; }
            </style>
        </head>
        <body>
            <h1>Face Recognition Attendance System</h1>
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
            <h2>Attendance Log</h2>
            <pre>{{ log }}</pre>
        </body>
    </html>
    """, log=open(attendance_file, "r").read() if os.path.exists(attendance_file) else "No logs yet.")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)