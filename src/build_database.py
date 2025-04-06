import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace

# Paths (relative to src/)
processed_dataset_path = "../processed_dataset/"
output_db_path = "../known_faces.pkl"

# Dictionary to store embeddings
embeddings_dict = {}

# Process each batchmate's folder
for batchmate in os.listdir(processed_dataset_path):
    batchmate_path = os.path.join(processed_dataset_path, batchmate)
    embeddings = []

    # Process each preprocessed image
    for img_file in os.listdir(batchmate_path):
        img_path = os.path.join(batchmate_path, img_file)
        img = cv2.imread(img_path)

        try:
            # Extract embedding using DeepFace (FaceNet model)
            embedding = DeepFace.represent(img, model_name="Facenet", enforce_detection=False)
            embeddings.append(embedding[0]["embedding"])  # DeepFace returns a list of dicts
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Average embeddings for this batchmate
    if embeddings:
        embeddings_dict[batchmate] = np.mean(embeddings, axis=0)
        print(f"Processed {batchmate} with {len(embeddings)} images")
    else:
        print(f"No valid embeddings for {batchmate}")

# Save to a DataFrame and pickle file
df = pd.DataFrame(embeddings_dict.items(), columns=["Name", "Embedding"])
df.to_pickle(output_db_path)

print(f"Database saved to {output_db_path}")