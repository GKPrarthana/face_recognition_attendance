from flask import Flask, Response, render_template_string, send_file, request
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import csv
from datetime import datetime, time
import os
from collections import defaultdict
import io

app = Flask(__name__)

# Paths
known_faces_path = "../known_faces.pkl"
attendance_file = "../attendance.csv"

# Load the known faces database
df = pd.read_pickle(known_faces_path)
known_embeddings = df["Embedding"].tolist()
known_names = df["Name"].tolist()

# Lecture slots
SLOT_1_START = time(8, 0)
SLOT_1_END = time(11, 0)
SLOT_1_LATE = time(8, 5)
SLOT_2_START = time(11, 30)
SLOT_2_END = time(14, 30)
SLOT_2_LATE = time(11, 35)
LATE_SOUND_CUTOFF = time(9, 30)

# Track logs per slot per student
logged_today = defaultdict(lambda: defaultdict(set))

# Store the latest detected name for marking
latest_name = "Unknown"

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def get_lecture_slot(current_time):
    current_t = current_time.time()
    if SLOT_1_START <= current_t <= SLOT_1_END:
        return "Slot 1 (8:00-11:00)"
    elif SLOT_2_START <= current_t <= SLOT_2_END:
        return "Slot 2 (11:30-14:30)"
    return None

def is_on_time(current_time, slot):
    current_t = current_time.time()
    if slot == "Slot 1 (8:00-11:00)":
        return current_t <= SLOT_1_LATE
    elif slot == "Slot 2 (11:30-14:30)":
        return current_t <= SLOT_2_LATE
    return False

def gen_frames():
    global latest_name
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Webcam Error", (200, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        try:
            frame = cv2.resize(frame, (640, 480))
            result = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True, detector_backend="mtcnn")
            embedding = result[0]["embedding"]
            face_coords = result[0]["facial_area"]

            distances = [np.linalg.norm(embedding - known_emb) for known_emb in known_embeddings]
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            threshold = 2.5
            name = known_names[min_distance_idx] if min_distance < threshold else "Unknown"
            latest_name = name  # Update the latest detected name

            x, y, w, h = face_coords["x"], face_coords["y"], face_coords["w"], face_coords["h"]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} ({min_distance:.2f})", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            slot = get_lecture_slot(datetime.now())
            slot_display = slot if slot else "Outside lecture hours"
            cv2.putText(frame, f"Slot: {slot_display}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        except Exception as e:
            print(f"Detection failed: {str(e)}")
            cv2.putText(frame, "No face detected", (200, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            latest_name = "Unknown"

        ret, buffer = cv2.imencode(".jpg", frame)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    log = open(attendance_file, "r").read() if os.path.exists(attendance_file) else "No logs yet."
    late_count = sum(1 for line in log.splitlines() if "Late" in line and "Slot 1" in line and "9:30" <= line.split(",")[1].split()[1][:5] <= "11:00")

    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Attendance Dashboard</title>
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                background: linear-gradient(to right, #2c3e50, #34495e);
                color: #ecf0f1;
                margin: 0;
                padding: 40px 60px;
            }
            h1, h2 {
                margin-bottom: 20px;
                letter-spacing: 1px;
            }
            .container {
                display: grid;
                grid-template-columns: 1fr 1.2fr;
                gap: 40px;
                max-width: 1600px;
                margin: 0 auto;
                align-items: flex-start;
            }
            .video-container, .dashboard {
                background: #3c5972;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
            }
            .video-feed {
                width: 100%;
                border: 5px solid #1abc9c;
                border-radius: 12px;
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.5);
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            th, td {
                padding: 14px 18px;
                border: 1px solid #7f8c8d;
                text-align: left;
                font-size: 15px;
            }
            th {
                background: #2980b9;
                position: sticky;
                top: 0;
                z-index: 1;
            }
            td {
                background: rgba(255, 255, 255, 0.03);
            }
            .on-time { color: #2ecc71; font-weight: 600; }
            .late { color: #e74c3c; font-weight: 600; }
            .late-alert {
                position: fixed;
                top: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: #e74c3c;
                color: white;
                padding: 14px 28px;
                border-radius: 8px;
                display: none;
                font-size: 18px;
                font-weight: bold;
                animation: pulse 1s ease-in-out infinite;
                box-shadow: 0 8px 18px rgba(0, 0, 0, 0.5);
                z-index: 9999;
            }
            @keyframes pulse {
                0%, 100% { transform: translateX(-50%) scale(1); }
                50% { transform: translateX(-50%) scale(1.05); }
            }
            button {
                background: linear-gradient(to right, #3498db, #1abc9c);
                color: white;
                border: none;
                padding: 14px 28px;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                margin-top: 25px;
                transition: background 0.3s ease, transform 0.2s ease;
            }
            button:hover {
                background: linear-gradient(to right, #1abc9c, #3498db);
                transform: scale(1.05);
            }
            #mark-attendance:disabled {
                background: #7f8c8d;
                cursor: not-allowed;
            }
            .summary {
                font-size: 17px;
                margin-top: 20px;
                font-weight: 500;
            }
            ::-webkit-scrollbar {
                width: 10px;
            }
            ::-webkit-scrollbar-thumb {
                background: #95a5a6;
                border-radius: 10px;
            }
        </style>
    </head>
    <body>
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="font-size: 36px; line-height: 1.4; margin: 0;">
                Attendance System
            </h1>
            <p style="margin: 10px 0 0; font-size: 20px; font-weight: 500;">
                Kotelawala Defence University
            </p>
            <p style="margin: 5px 0 0; font-size: 20px; color: #1abc9c; font-weight: 500;">
                BSc (Hons) Data Science & Business Analysis
            </p>
            <p style="margin: 5px 0 0; font-size: 18px; color: #ecf0f1;">
                Intake 39
            </p>
        </div>
        
        <div class="container">
            <div class="video-container">
            
                <img src="{{ url_for('video_feed') }}" class="video-feed" alt="Live Video Feed">
                <button id="mark-attendance" onclick="markAttendance()"> Mark Attendance</button>
            </div>
            <div class="dashboard">
                <h2>Attendance Log</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Timestamp</th>
                            <th>Status</th>
                            <th>Slot</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for line in log.splitlines()[1:] %}
                            {% set parts = line.split(',') %}
                            {% if parts|length >= 4 %}
                                <tr>
                                    <td>{{ parts[0] }}</td>
                                    <td>{{ parts[1] }}</td>
                                    <td class="{{ 'on-time' if parts[2] == 'On Time' else 'late' }}">{{ parts[2] }}</td>
                                    <td>{{ parts[3] }}</td>
                                </tr>
                            {% endif %}
                        {% endfor %}
                    </tbody>
                </table>
                <p class="summary">👥 Total Latecomers after 9:30 AM: <strong>{{ late_count }}</strong></p>
                <button onclick="window.location.href='/download'">⬇ Download attendance</button>
            </div>
        </div>

        <div id="late-alert" class="late-alert">⚠️ Latecomer After 9:30!</div>

        <script>
            function playLateSound() {
                const audio = new Audio('https://www.soundjay.com/buttons/beep-07.wav');
                audio.play();
                const alert = document.getElementById('late-alert');
                alert.style.display = 'block';
                setTimeout(() => { alert.style.display = 'none'; }, 3000);
            }

            function markAttendance() {
                fetch('/mark_attendance', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            location.reload();  // Refresh to update log
                        } else {
                            alert(data.message);
                        }
                    });
            }

            setInterval(() => {
                fetch('/check_late')
                    .then(response => response.json())
                    .then(data => { if (data.late) playLateSound(); });
            }, 1000);

            setInterval(() => {
                fetch('/get_latest_name')
                    .then(response => response.json())
                    .then(data => {
                        const button = document.getElementById('mark-attendance');
                        button.disabled = (data.name === 'Unknown');
                    });
            }, 500);
        </script>
    </body>
    </html>
    """, log=log, late_count=late_count)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_late')
def check_late():
    current_time = datetime.now()
    slot = get_lecture_slot(current_time)
    if slot == "Slot 1 (8:00-11:00)" and current_time.time() >= LATE_SOUND_CUTOFF:
        with open(attendance_file, "r") as f:
            last_line = f.readlines()[-1].strip()
            if "Late" in last_line and last_line not in logged_today[current_time.strftime("%Y-%m-%d")].get("last_check", ""):
                logged_today[current_time.strftime("%Y-%m-%d")]["last_check"] = last_line
                return {"late": True}
    return {"late": False}

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    global latest_name
    if latest_name == "Unknown":
        return {"success": False, "message": "No student identified. Please ensure your face is recognized."}
    
    current_time = datetime.now()
    current_date = current_time.strftime("%Y-%m-%d")
    slot = get_lecture_slot(current_time)
    
    if not slot:
        return {"success": False, "message": "Outside lecture hours. Attendance cannot be marked."}
    
    if slot in logged_today[current_date][latest_name]:
        return {"success": False, "message": f"Attendance already marked for {latest_name} in {slot}."}
    
    status = "On Time" if is_on_time(current_time, slot) else "Late"
    with open(attendance_file, "a", newline="") as f:
        writer = csv.writer(f)
        if os.path.getsize(attendance_file) == 0:
            writer.writerow(["Name", "Timestamp", "Status", "Lecture Slot"])
        writer.writerow([latest_name, current_time.strftime("%Y-%m-%d %H:%M:%S"), status, slot])
    logged_today[current_date][latest_name].add(slot)
    return {"success": True, "message": "Attendance marked successfully!"}

@app.route('/get_latest_name')
def get_latest_name():
    return {"name": latest_name}

@app.route('/download')
def download_csv():
    with open(attendance_file, "rb") as f:
        return send_file(io.BytesIO(f.read()), mimetype="text/csv", as_attachment=True, download_name="attendance.csv")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)