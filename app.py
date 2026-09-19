import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import av

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("😊 Facial Emotion Recognition")

st.write(
    "CNN-based Facial Emotion Recognition using "
    "FER-2013, OpenCV and Streamlit."
)

st.divider()


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "emotion_model.keras"
    )


model = load_model()


# ============================================================
# EMOTION CLASSES
# ============================================================

class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# ============================================================
# FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# COMMON PREDICTION FUNCTION
# ============================================================

def predict_emotion(face):

    # Resize face
    face = cv2.resize(
        face,
        (48, 48)
    )

    # Normalize pixel values
    face = face.astype(
        np.float32
    ) / 255.0

    # Add channel dimension
    face = np.expand_dims(
        face,
        axis=-1
    )

    # Add batch dimension
    face = np.expand_dims(
        face,
        axis=0
    )

    # Model prediction
    predictions = model.predict(
        face,
        verbose=0
    )[0]

    # Get highest probability
    predicted_index = np.argmax(
        predictions
    )

    emotion = class_names[
        predicted_index
    ]

    confidence = (
        predictions[predicted_index] * 100
    )

    return emotion, confidence


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Prediction Mode")

mode = st.sidebar.radio(
    "Choose an option:",
    [
        "🎥 Real-Time Prediction",
        "📸 Webcam Picture Prediction",
        "🖼️ Upload Picture Prediction"
    ]
)


st.sidebar.divider()

st.sidebar.subheader("📌 About")

st.sidebar.write(
    "This application uses a CNN trained on "
    "the FER-2013 dataset to recognize facial emotions."
)

st.sidebar.write("**Recognized Emotions:**")

for emotion in class_names:

    st.sidebar.write(
        f"• {emotion}"
    )


# ============================================================
# OPTION 1
# REAL-TIME WEBCAM PREDICTION
# ============================================================

if mode == "🎥 Real-Time Prediction":

    st.header("🎥 Real-Time Emotion Prediction")

    st.write(
        "Start the webcam to detect and classify "
        "facial emotions continuously."
    )


    class EmotionProcessor(
        VideoProcessorBase
    ):

        def recv(self, frame):

            # Convert frame to OpenCV format
            img = frame.to_ndarray(
                format="bgr24"
            )

            # Mirror webcam
            img = cv2.flip(
                img,
                1
            )

            # Convert to grayscale
            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )


            # Process every detected face
            for (x, y, w, h) in faces:

                # Crop face
                face = gray[
                    y:y+h,
                    x:x+w
                ]

                # Predict
                emotion, confidence = (
                    predict_emotion(face)
                )


                # Draw face rectangle
                cv2.rectangle(
                    img,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )


                # Prediction text
                text = (
                    f"{emotion}: "
                    f"{confidence:.1f}%"
                )


                # Display prediction
                cv2.putText(
                    img,
                    text,
                    (
                        x,
                        max(y - 10, 30)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


            # Return processed frame
            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24"
            )


    # Start WebRTC webcam
    webrtc_streamer(
        key="real-time-emotion",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )


# ============================================================
# OPTION 2
# WEBCAM PICTURE PREDICTION
# ============================================================

elif mode == "📸 Webcam Picture Prediction":

    st.header("📸 Webcam Picture Prediction")

    st.write(
        "Take a single picture using your webcam "
        "and predict the facial emotion."
    )


    # Webcam picture
    camera_image = st.camera_input(
        "Take a picture"
    )


    if camera_image is not None:

        # Read image
        image_bytes = np.asarray(
            bytearray(
                camera_image.getvalue()
            ),
            dtype=np.uint8
        )


        image = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )


        # Convert to grayscale
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )


        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )


        st.write(
            f"**Faces detected: {len(faces)}**"
        )


        if len(faces) == 0:

            st.warning(
                "No face detected. "
                "Please take another picture "
                "with your face clearly visible."
            )


        # Process detected faces
        for (x, y, w, h) in faces:

            face = gray[
                y:y+h,
                x:x+w
            ]


            emotion, confidence = (
                predict_emotion(face)
            )


            # Draw rectangle
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )


            # Prediction label
            text = (
                f"{emotion}: "
                f"{confidence:.1f}%"
            )


            cv2.putText(
                image,
                text,
                (
                    x,
                    max(y - 10, 30)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        # Convert BGR to RGB
        result = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        st.subheader(
            "🔎 Prediction Result"
        )


        st.image(
            result,
            caption="Webcam Picture Prediction",
            use_container_width=True
        )


# ============================================================
# OPTION 3
# UPLOAD PICTURE PREDICTION
# ============================================================

elif mode == "🖼️ Upload Picture Prediction":

    st.header("🖼️ Upload Picture Prediction")

    st.write(
        "Upload a JPG, JPEG or PNG image "
        "to predict the facial emotion."
    )


    # Upload image
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        # Read uploaded image
        image_bytes = np.asarray(
            bytearray(
                uploaded_file.read()
            ),
            dtype=np.uint8
        )


        image = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )


        if image is None:

            st.error(
                "Unable to read the image."
            )

        else:

            # Convert to grayscale
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )


            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )


            st.write(
                f"**Faces detected: {len(faces)}**"
            )


            if len(faces) == 0:

                st.warning(
                    "No face detected. "
                    "Please upload a clearer "
                    "face image."
                )


            # Process each face
            for (x, y, w, h) in faces:

                # Crop face
                face = gray[
                    y:y+h,
                    x:x+w
                ]


                # Predict emotion
                emotion, confidence = (
                    predict_emotion(face)
                )


                # Draw rectangle
                cv2.rectangle(
                    image,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    3
                )


                # Prediction text
                text = (
                    f"{emotion}: "
                    f"{confidence:.1f}%"
                )


                cv2.putText(
                    image,
                    text,
                    (
                        x,
                        max(y - 10, 30)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


            # Convert BGR to RGB
            result = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )


            st.subheader(
                "🔎 Prediction Result"
            )


            st.image(
                result,
                caption="Uploaded Image Prediction",
                use_container_width=True
            )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("📊 Project Information")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.write("**Dataset**")
    st.write("FER-2013")


with col2:

    st.write("**Model**")
    st.write("CNN")


with col3:

    st.write("**Input**")
    st.write("48 × 48 Grayscale")


with col4:

    st.write("**Classes**")
    st.write("7")


st.divider()

st.write(
    "OpenCV detects the face, the face is converted "
    "to a 48 × 48 grayscale image, and the trained "
    "CNN predicts one of seven facial emotions."
)