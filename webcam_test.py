import cv2
import numpy as np
import tensorflow as tf

# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

model = tf.keras.models.load_model("emotion_model.keras")

# Emotion labels
class_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# ============================================================
# 2. LOAD FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ============================================================
# 3. OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Webcam opened successfully!")
print("Press Q to quit.")

# ============================================================
# 4. REAL-TIME LOOP
# ============================================================

while True:

    # Capture frame
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    # Convert webcam frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # ========================================================
    # 5. PROCESS EACH FACE
    # ========================================================

    for (x, y, w, h) in faces:

        # Crop face
        face = gray[y:y+h, x:x+w]

        # Resize to model input size
        face = cv2.resize(face, (48, 48))

        # Normalize pixel values
        face = face / 255.0

        # Add channel dimension
        face = np.expand_dims(face, axis=-1)

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        # ====================================================
        # 6. PREDICT EMOTION
        # ====================================================

        predictions = model.predict(face, verbose=0)

        predicted_index = np.argmax(predictions[0])

        predicted_emotion = class_names[predicted_index]

        confidence = predictions[0][predicted_index] * 100

        # ====================================================
        # 7. DRAW FACE RECTANGLE
        # ====================================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # ====================================================
        # 8. DISPLAY EMOTION
        # ====================================================

        text = f"{predicted_emotion}: {confidence:.1f}%"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # ========================================================
    # 9. DISPLAY WEBCAM
    # ========================================================

    cv2.imshow("Real-Time Facial Emotion Recognition", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# 10. CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()