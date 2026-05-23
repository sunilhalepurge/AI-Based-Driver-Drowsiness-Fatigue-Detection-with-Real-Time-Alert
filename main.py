import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp
import numpy as np
import pygame

# ================== LOAD MODEL ==================
model = load_model("models/eye_model.h5", compile=False)
# ================== INIT ALARM ==================
pygame.mixer.init()

# Safe path handling
base_dir = os.path.dirname(os.path.abspath(__file__))
alarm_path = os.path.join(base_dir, "alarm.wav")

if not os.path.exists(alarm_path):
    print("❌ ERROR: alarm.wav not found at:", alarm_path)
else:
    print("✅ Alarm loaded from:", alarm_path)
    pygame.mixer.music.load(alarm_path)

# ================== MEDIAPIPE ==================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ================== START CAMERA ==================
cap = cv2.VideoCapture(0)

alarm_on = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = frame.shape

            # Get eye coordinates
            left_eye = []
            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye.append((x, y))

            # Bounding box
            x_coords = [p[0] for p in left_eye]
            y_coords = [p[1] for p in left_eye]

            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)

            eye = frame[y1:y2, x1:x2]

            if eye.size != 0:
                eye = cv2.resize(eye, (24, 24))
                eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
                eye = eye / 255.0
                eye = np.reshape(eye, (1, 24, 24, 1))

                prediction = model.predict(eye, verbose=0)

                if prediction[0][0] > 0.5:
                    text = "Closed Eyes 😴"
                    color = (0, 0, 255)

                    if not alarm_on:
                        try:
                            pygame.mixer.music.play()
                            alarm_on = True
                        except:
                            print("⚠️ Alarm play error")

                else:
                    text = "Open Eyes 👀"
                    color = (0, 255, 0)
                    pygame.mixer.music.stop()
                    alarm_on = False

                cv2.putText(frame, text, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()