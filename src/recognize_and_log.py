import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import csv
from datetime import datetime
import os
from collections import defaultdict

# Paths
known_faces_path = "../known_faces.pkl"
attendance_file = "../attendance.csv"

# Load the known faces database
df = pd.read_pickle(known_faces_path)
known_embeddings = df["Embedding"].tolist()
known_names = df["Name"].tolist()

# Cooldown dictionary
cooldown_period = 300
last_logged = defaultdict(lambda: datetime.min)
attendance_count = 0

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    try:
        # Extract embedding and face coordinates
        result = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True, detector_backend="opencv")
        embedding = result[0]["embedding"]
        face_coords = result[0]["facial_area"]

        # Compare with known embeddings
        distances = [np.linalg.norm(embedding - known_emb) for known_emb in known_embeddings]
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        threshold = 1.2
        name = known_names[min_distance_idx] if min_distance < threshold else "Unknown"

        # Debug info
        print(f"Closest match: {known_names[min_distance_idx]}, Distance: {min_distance:.2f}")

        # Draw bounding box and label
        x, y, w, h = face_coords["x"], face_coords["y"], face_coords["w"], face_coords["h"]
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{name} ({min_distance:.2f})", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Log attendance
        current_time = datetime.now()
        if name != "Unknown" and (current_time - last_logged[name]).total_seconds() > cooldown_period:
            with open(attendance_file, "a", newline="") as f:
                writer = csv.writer(f)
                if os.path.getsize(attendance_file) == 0:
                    writer.writerow(["Name", "Timestamp"])
                writer.writerow([name, current_time.strftime("%Y-%m-%d %H:%M:%S")])
            print(f"Attendance logged for {name}")
            last_logged[name] = current_time
            attendance_count += 1

        # Display attendance count
        cv2.putText(frame, f"Attendance: {attendance_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    except Exception as e:
        print(f"Detection failed: {str(e)}")
        cv2.putText(frame, "No face detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\nAttendance Summary:")
with open(attendance_file, "r") as f:
    print(f.read())