# capture_faces.py
import cv2
import os

person_name = input("Enter person's name: ").lower().replace(" ", "_")
num_samples = 30
path = f"dataset/{person_name}"
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
print(f"Capturing {num_samples} photos of {person_name}... Look at camera!")

while count < num_samples:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_roi = gray[y:y+h, x:x+w]
        resized = cv2.resize(face_roi, (200, 200))
        cv2.imwrite(f"{path}/{count}.jpg", resized)
        count += 1

    cv2.putText(frame, f"Captured: {count}/{num_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Capturing Face", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done! Saved {count} images for {person_name}")