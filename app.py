import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 28px;
    font-weight: 650;
    margin-top: 25px;
    margin-bottom: 15px;
}

.info-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.3);
    margin-bottom: 15px;
}

.emotion-card {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.3);
    text-align: center;
    margin-bottom: 10px;
}

.small-text {
    font-size: 15px;
    line-height: 1.6;
}

.footer {
    text-align: center;
    padding: 20px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# EMOTION CLASSES
# ============================================================

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
# LOAD TFLITE MODEL
# ============================================================

@st.cache_resource
def load_model():

    interpreter = Interpreter(
        model_path="emotion_model.tflite"
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


interpreter, input_details, output_details = load_model()


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

@st.cache_resource
def load_face_detector():

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    return detector


face_cascade = load_face_detector()


# ============================================================
# EMOTION PREDICTION FUNCTION
# ============================================================

def predict_emotion(face):

    # Resize face to 48 × 48
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

    # Send image to model
    interpreter.set_tensor(
        input_details[0]["index"],
        face
    )

    # Run inference
    interpreter.invoke()

    # Get prediction probabilities
    predictions = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    # Get predicted class
    predicted_index = np.argmax(
        predictions
    )

    predicted_emotion = class_names[
        predicted_index
    ]

    # Confidence percentage
    confidence = (
        predictions[predicted_index] * 100
    )

    return predicted_emotion, confidence


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("😊 Facial Emotion Recognition")

st.sidebar.markdown(
    "### 🎯 Prediction Mode"
)

mode = st.sidebar.radio(
    "Choose an option:",
    [
        "🏠 About Project",
        "🎥 Real-Time Prediction",
        "📸 Webcam Picture Prediction",
        "🖼️ Upload Picture Prediction"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### 🧠 Model Information

    **Model:** CNN

    **Dataset:** FER-2013

    **Input:** 48 × 48 Grayscale

    **Classes:** 7

    **Face Detection:** Haar Cascade

    **Deployment:** LiteRT / TFLite
    """
)


# ============================================================
# ABOUT PROJECT
# ============================================================

if mode == "🏠 About Project":

    st.markdown(
        '<div class="main-title">😊 Facial Emotion Recognition</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'CNN-Based Facial Emotion Recognition System'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🧠 About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card small-text">

        Facial Emotion Recognition is a computer vision and
        deep learning project that identifies human emotions
        from facial images.

        The system uses a Convolutional Neural Network (CNN)
        trained on the FER-2013 dataset. The model recognizes
        seven different facial expressions:

        <b>Angry, Disgust, Fear, Happy, Neutral, Sad and Surprise.</b>

        OpenCV is used to detect faces from images and webcam
        frames. The detected face is converted into grayscale,
        resized to 48 × 48 pixels and normalized before being
        passed to the trained CNN model.

        The application provides real-time webcam prediction,
        webcam picture prediction and uploaded-image prediction.

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # PROJECT INFORMATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Project Information</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="info-card">

            <b>📂 Dataset</b><br>
            FER-2013

            <br><br>

            <b>🖼️ Image Size</b><br>
            48 × 48 pixels

            <br><br>

            <b>🎨 Image Type</b><br>
            Grayscale

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="info-card">

            <b>🤖 Model</b><br>
            Convolutional Neural Network

            <br><br>

            <b>😊 Classes</b><br>
            7 Emotions

            <br><br>

            <b>👤 Face Detection</b><br>
            Haar Cascade

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="info-card">

            <b>🐍 Language</b><br>
            Python

            <br><br>

            <b>🧠 Deep Learning</b><br>
            TensorFlow / Keras

            <br><br>

            <b>🌐 Interface</b><br>
            Streamlit

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # EMOTION CLASSES
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">😊 Emotion Classes</div>',
        unsafe_allow_html=True
    )

    emotion_col1, emotion_col2, emotion_col3, emotion_col4 = st.columns(4)

    with emotion_col1:
        st.markdown(
            '<div class="emotion-card">😠<br><b>Angry</b></div>',
            unsafe_allow_html=True
        )

    with emotion_col2:
        st.markdown(
            '<div class="emotion-card">🤢<br><b>Disgust</b></div>',
            unsafe_allow_html=True
        )

    with emotion_col3:
        st.markdown(
            '<div class="emotion-card">😨<br><b>Fear</b></div>',
            unsafe_allow_html=True
        )

    with emotion_col4:
        st.markdown(
            '<div class="emotion-card">😊<br><b>Happy</b></div>',
            unsafe_allow_html=True
        )

    emotion_col5, emotion_col6, emotion_col7 = st.columns(3)

    with emotion_col5:
        st.markdown(
            '<div class="emotion-card">😐<br><b>Neutral</b></div>',
            unsafe_allow_html=True
        )

    with emotion_col6:
        st.markdown(
            '<div class="emotion-card">😢<br><b>Sad</b></div>',
            unsafe_allow_html=True
        )

    with emotion_col7:
        st.markdown(
            '<div class="emotion-card">😲<br><b>Surprise</b></div>',
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🔄 How the System Works</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">

        <b>Step 1 — Image / Webcam Input</b><br>
        The system receives an image or webcam frame.

        <br><br>

        ↓

        <br><br>

        <b>Step 2 — Face Detection</b><br>
        OpenCV Haar Cascade detects the face.

        <br><br>

        ↓

        <br><br>

        <b>Step 3 — Face Preprocessing</b><br>
        The detected face is converted to grayscale and resized
        to 48 × 48 pixels.

        <br><br>

        ↓

        <br><br>

        <b>Step 4 — Normalization</b><br>
        Pixel values are converted from 0–255 to 0–1.

        <br><br>

        ↓

        <br><br>

        <b>Step 5 — CNN Prediction</b><br>
        The trained CNN analyzes the facial features.

        <br><br>

        ↓

        <br><br>

        <b>Step 6 — Emotion Result</b><br>
        The system displays the predicted emotion and
        confidence percentage.

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # TECHNOLOGIES
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🛠️ Technologies Used</div>',
        unsafe_allow_html=True
    )

    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:

        st.markdown(
            """
            <div class="info-card">

            🐍 <b>Python</b> — Programming language

            <br><br>

            🧠 <b>TensorFlow / Keras</b> — CNN model development

            <br><br>

            👁️ <b>OpenCV</b> — Image processing and face detection

            <br><br>

            🔢 <b>NumPy</b> — Numerical operations

            </div>
            """,
            unsafe_allow_html=True
        )

    with tech_col2:

        st.markdown(
            """
            <div class="info-card">

            🌐 <b>Streamlit</b> — Web application

            <br><br>

            ⚡ <b>LiteRT / TFLite</b> — Lightweight model inference

            <br><br>

            📷 <b>WebRTC</b> — Real-time webcam streaming

            <br><br>

            📊 <b>FER-2013</b> — Facial expression dataset

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">⚠️ Limitations</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card small-text">

        • Prediction accuracy can be affected by poor lighting.

        <br><br>

        • Face angle and occlusion can affect detection.

        <br><br>

        • Facial expressions do not always perfectly represent
        a person's actual emotional state.

        <br><br>

        • The model is trained on the FER-2013 dataset and may
        perform differently on real-world images.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# REAL-TIME PREDICTION
# ============================================================

elif mode == "🎥 Real-Time Prediction":

    st.markdown(
        '<div class="main-title">🎥 Real-Time Emotion Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="small-text">

        Allow camera access to detect facial emotions
        continuously in real time.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    class EmotionProcessor(VideoProcessorBase):

        def recv(self, frame):

            img = frame.to_ndarray(
                format="bgr24"
            )

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:

                face = gray[
                    y:y + h,
                    x:x + w
                ]

                try:

                    emotion, confidence = predict_emotion(
                        face
                    )

                    cv2.rectangle(
                        img,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

                    text = (
                        f"{emotion}: "
                        f"{confidence:.1f}%"
                    )

                    cv2.putText(
                        img,
                        text,
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                except Exception:
                    pass

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24"
            )

    webrtc_streamer(
        key="emotion-recognition",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )


# ============================================================
# WEBCAM PICTURE PREDICTION
# ============================================================

elif mode == "📸 Webcam Picture Prediction":

    st.markdown(
        '<div class="main-title">📸 Webcam Picture Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Take a picture using your webcam to detect facial emotion."
    )

    st.markdown("---")

    camera_image = st.camera_input(
        "Take a picture"
    )

    if camera_image is not None:

        image = Image.open(
            camera_image
        ).convert("RGB")

        image_array = np.array(
            image
        )

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        result_image = image_array.copy()

        if len(faces) == 0:

            st.warning(
                "No face detected. Please try another picture."
            )

            st.image(
                image,
                caption="Captured Image",
                use_container_width=True
            )

        else:

            st.success(
                f"{len(faces)} face(s) detected."
            )

            for (x, y, w, h) in faces:

                face = gray[
                    y:y + h,
                    x:x + w
                ]

                emotion, confidence = predict_emotion(
                    face
                )

                cv2.rectangle(
                    result_image,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                text = (
                    f"{emotion}: "
                    f"{confidence:.1f}%"
                )

                cv2.putText(
                    result_image,
                    text,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            st.image(
                result_image,
                caption="Facial Emotion Prediction",
                use_container_width=True
            )


# ============================================================
# UPLOAD IMAGE PREDICTION
# ============================================================

elif mode == "🖼️ Upload Picture Prediction":

    st.markdown(
        '<div class="main-title">🖼️ Upload Picture Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload an image containing one or more faces."
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_array = np.array(
            image
        )

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        result_image = image_array.copy()

        if len(faces) == 0:

            st.warning(
                "No face detected in the uploaded image."
            )

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        else:

            st.success(
                f"{len(faces)} face(s) detected."
            )

            for (x, y, w, h) in faces:

                face = gray[
                    y:y + h,
                    x:x + w
                ]

                emotion, confidence = predict_emotion(
                    face
                )

                cv2.rectangle(
                    result_image,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                text = (
                    f"{emotion}: "
                    f"{confidence:.1f}%"
                )

                cv2.putText(
                    result_image,
                    text,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            st.image(
                result_image,
                caption="Facial Emotion Prediction",
                use_container_width=True
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    <b>Facial Emotion Recognition</b><br>

    CNN • OpenCV • FER-2013 • Streamlit • LiteRT

    </div>
    """,
    unsafe_allow_html=True
)