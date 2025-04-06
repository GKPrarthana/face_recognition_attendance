import cv2
import os
import numpy as np

# Load OpenCV's pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Paths (relative to project root)
input_dataset_path = "C:/Users/Prarthana/Downloads/dba_39"
output_dataset_path = "../processed_dataset"

# Create output directory if it doesn't exist
if not os.path.exists(output_dataset_path):
    os.makedirs(output_dataset_path)

# Process each batchmate's folder
for batchmate in os.listdir(input_dataset_path):
    batchmate_path = os.path.join(input_dataset_path, batchmate)
    output_batchmate_path = os.path.join(output_dataset_path, batchmate)
    
    if not os.path.exists(output_batchmate_path):
        os.makedirs(output_batchmate_path)

    # Process each image
    for img_file in os.listdir(batchmate_path):
        img_path = os.path.join(batchmate_path, img_file)
        img = cv2.imread(img_path)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Crop and save each detected face
        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y+h, x:x+w]  # Crop the face
            face_resized = cv2.resize(face, (224, 224))  # Resize to 224x224
            output_path = os.path.join(output_batchmate_path, f"{img_file.split('.')[0]}_face{i}.jpg")
            cv2.imwrite(output_path, face_resized)

print("Preprocessing complete! Check the 'processed_dataset' folder.")