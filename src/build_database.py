import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace

# Paths
processed_dataset_path = "../processed_dataset/"
output_db_path = "../known_faces.pkl"

# Dictionary to hold average embeddings for each person
embeddings_dict = {}

print("📦 Building face embeddings database...")

for person in os.listdir(processed_dataset_path):
    person_path = os.path.join(processed_dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    embeddings = []
    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Could not read image: {img_path}")
            continue

        try:
            # Generate face embedding
            result = DeepFace.represent(img, model_name="Facenet", enforce_detection=False)
            embeddings.append(result[0]["embedding"])
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            continue

    if embeddings:
        embeddings_dict[person] = np.mean(embeddings, axis=0)
        print(f"✅ Processed {person} with {len(embeddings)} valid images")
    else:
        print(f"⚠️ No valid embeddings for {person}")

# Save the database
df = pd.DataFrame(embeddings_dict.items(), columns=["Name", "Embedding"])
df.to_pickle(output_db_path)
print(f"\n🎉 Database saved successfully to: {output_db_path}")
