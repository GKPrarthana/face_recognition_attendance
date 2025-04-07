import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import csv
from datetime import datetime, time
import os
from collections import defaultdict
import winsound
import time as time_module  # To add a pause between beeps

# Paths
known_faces_path = "../known_faces.pkl"
attendance_file = "../attendance.csv"

# Load the known faces database
df = pd.read_pickle(known_faces_path)
known_embeddings = df["Embedding"].tolist()
known_names = df["Name"].tolist()

# Lecture slots (in 24-hour format)
SLOT_1_START = time(8, 0)    # 8:00 AM
SLOT_1_END = time(11, 0)     # 11:00 AM
SLOT_1_LATE = time(8, 5)     # 8:05 AM
SLOT_2_START = time(11, 30)  # 11:30 AM
SLOT_2_END = time(14, 30)    # 2:30 PM
SLOT_2_LATE = time(11, 35)   # 11:35 AM
LATE_SOUND_CUTOFF = time(9, 30)  # 9:30 AM for sound alert

# Track logs per slot per student (date -> name -> slot)
logged_today = defaultdict(lambda: defaultdict(set))

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam... Press 'q' to quit.")

def get_lecture_slot(current_time):
    """Determine the current lecture slot based on time."""
    current_t = current_time.time()
    if SLOT_1_START <= current_t <= SLOT_1_END:
        return "Slot 1 (8:00-11:00)"
    elif SLOT_2_START <= current_t <= SLOT_2_END:
        return "Slot 2 (11:30-14:30)"
    return None

def is_on_time(current_time, slot):
    """Check if the student is on time for the slot."""
    current_t = current_time.time()
    if slot == "Slot 1 (8:00-11:00)":
        return current_t <= SLOT_1_LATE
    elif slot == "Slot 2 (11:30-14:30)":
        return current_t <= SLOT_2_LATE
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    try:
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Extract embedding and face coordinates using OpenCV detector
        result = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True, detector_backend="mtcnn")
        embedding = result[0]["embedding"]
        face_coords = result[0]["facial_area"]

        # Compare with known embeddings
        distances = [np.linalg.norm(embedding - known_emb) for known_emb in known_embeddings]
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        threshold = 1.2  # Adjust if needed
        name = known_names[min_distance_idx] if min_distance < threshold else "Unknown"

        # Debug info
        print(f"Closest match: {known_names[min_distance_idx]}, Distance: {min_distance:.2f}, Threshold: {threshold}")

        # Draw bounding box and label
        x, y, w, h = face_coords["x"], face_coords["y"], face_coords["w"], face_coords["h"]
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{name} ({min_distance:.2f})", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Log attendance (one per slot)
        current_time = datetime.now()
        current_date = current_time.strftime("%Y-%m-%d")
        slot = get_lecture_slot(current_time)

        if name != "Unknown" and slot and slot not in logged_today[current_date][name]:
            status = "On Time" if is_on_time(current_time, slot) else "Late"
            with open(attendance_file, "a", newline="") as f:
                writer = csv.writer(f)
                if os.path.getsize(attendance_file) == 0:
                    writer.writerow(["Name", "Timestamp", "Status", "Lecture Slot"])
                writer.writerow([name, current_time.strftime("%Y-%m-%d %H:%M:%S"), status, slot])
            print(f"Attendance logged for {name}: {status} - {slot}")
            logged_today[current_date][name].add(slot)

            # Sound alert for latecomers after 9:30 AM in Slot 1
            if slot == "Slot 1 (8:00-11:00)" and status == "Late" and current_time.time() >= LATE_SOUND_CUTOFF:
                # Double beep: softer tone (500 Hz), 200 ms each, with a 100 ms pause
                winsound.Beep(500, 200)
                time_module.sleep(0.1)  # Pause between beeps
                winsound.Beep(500, 200)
                print("Latecomer alert sounded!")

        # Display current slot
        slot_display = slot if slot else "Outside lecture hours"
        cv2.putText(frame, f"Slot: {slot_display}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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