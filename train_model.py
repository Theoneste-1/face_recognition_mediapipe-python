# train_model.py
import cv2
import os
import numpy as np
import pickle
import json
from pathlib import Path

dataset_path = "dataset"
model_path = "models/lbph_model.yml"
label_path = "models/label_map.json"

faces = []
labels = []
label_dict = {}
current_id = 0

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue
    label_dict[current_id] = person_name
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(current_id)
    current_id += 1

# Train LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Save model and label map
os.makedirs("models", exist_ok=True)
recognizer.save(model_path)

with open(label_path, 'w') as f:
    json.dump(label_dict, f)

print("Training complete! Model saved.")
print("People trained:", label_dict)