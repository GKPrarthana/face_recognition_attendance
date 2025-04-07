import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Attempt face detection
        result = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True, detector_backend="opencv")
        face_coords = result[0]["facial_area"]
        x, y, w, h = face_coords["x"], face_coords["y"], face_coords["w"], face_coords["h"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    except Exception as e:
        cv2.putText(frame, f"No face detected: {str(e)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Face Detection Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()