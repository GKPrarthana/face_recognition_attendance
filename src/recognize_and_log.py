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

# Cooldown dictionary to prevent duplicate logs (in seconds)
cooldown_period = 300  # 5 minutes
last_logged = defaultdict(lambda: datetime.min)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is default camera; change if needed
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
        # Extract embedding from the current frame
        embedding = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True)[0]["embedding"]

        # Compare with known embeddings (Euclidean distance)
        distances = [np.linalg.norm(embedding - known_emb) for known_emb in known_embeddings]
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]

        # Threshold for recognition
        threshold = 0.6
        if min_distance < threshold:
            name = known_names[min_distance_idx]
        else:
            name = "Unknown"

        # Display result on frame
        cv2.putText(frame, f"{name} ({min_distance:.2f})", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Log attendance if not recently logged
        current_time = datetime.now()
        if name != "Unknown" and (current_time - last_logged[name]).total_seconds() > cooldown_period:
            with open(attendance_file, "a", newline="") as f:
                writer = csv.writer(f)
                if os.path.getsize(attendance_file) == 0:
                    writer.writerow(["Name", "Timestamp"])
                writer.writerow([name, current_time.strftime("%Y-%m-%d %H:%M:%S")])
            print(f"Attendance logged for {name}")
            last_logged[name] = current_time

    except Exception as e:
        # Skip if face detection or embedding fails (e.g., no face in frame)
        cv2.putText(frame, "No face detected", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Attendance System", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Webcam stopped. Attendance logged in attendance.csv.")