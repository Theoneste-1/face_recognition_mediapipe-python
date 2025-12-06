# recognize_live.py
import cv2
import json
import numpy as np

# Load model and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/lbph_model.yml")

with open("models/label_map.json", 'r') as f:
    label_map = json.load(f)
    # Reverse: id -> name
    id_to_name = {int(k): v for k, v in label_map.items()}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))

        label_id, confidence = recognizer.predict(face_roi)

        if confidence < 100:
            name = id_to_name.get(label_id, "Unknown")
            text = f"{name} ({confidence:.0f}%)"
            color = (0, 255, 0)
        else:
            text = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition - LBPH", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()