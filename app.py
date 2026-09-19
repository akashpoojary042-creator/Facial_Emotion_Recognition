import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Facial Emotion AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ---------- GENERAL ---------- */

.stApp {
    background: #f5f7fb;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}


/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"] {
    background: #111827;
}

section[data-testid="stSidebar"] * {
    color: #ffffff;
}

.sidebar-title {
    text-align: center;
    padding: 15px 5px 25px 5px;
}

.sidebar-title h1 {
    font-size: 28px;
    margin-bottom: 5px;
}

.sidebar-title p {
    color: #cbd5e1;
    font-size: 13px;
}


/* ---------- HERO ---------- */

.hero {
    background:
        linear-gradient(
            135deg,
            #4f46e5 0%,
            #7c3aed 50%,
            #9333ea 100%
        );

    padding: 42px 35px;
    border-radius: 24px;
    color: white;
    text-align: center;

    box-shadow:
        0 15px 35px rgba(79, 70, 229, 0.20);

    margin-bottom: 28px;
}

.hero-icon {
    font-size: 55px;
    margin-bottom: 5px;
}

.hero h1 {
    font-size: 44px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
}

.hero p {
    font-size: 18px;
    margin-top: 12px;
    color: #ede9fe;
}


/* ---------- SECTION TITLE ---------- */

.section-title {
    font-size: 28px;
    font-weight: 750;
    color: #111827;
    margin-top: 10px;
    margin-bottom: 18px;
}


/* ---------- INFORMATION CARD ---------- */

.info-card {
    background: white;
    padding: 22px;
    border-radius: 18px;

    border: 1px solid #e5e7eb;

    box-shadow:
        0 5px 18px rgba(0, 0, 0, 0.04);

    margin-bottom: 20px;
}


/* ---------- FEATURE CARDS ---------- */

.feature-card {
    background: white;
    padding: 25px 20px;
    border-radius: 18px;

    text-align: center;

    border: 1px solid #e5e7eb;

    box-shadow:
        0 5px 18px rgba(0, 0, 0, 0.04);

    min-height: 160px;
}

.feature-icon {
    font-size: 38px;
}

.feature-title {
    font-size: 18px;
    font-weight: 700;
    margin-top: 10px;
    color: #111827;
}

.feature-text {
    color: #6b7280;
    font-size: 14px;
    margin-top: 7px;
}


/* ---------- RESULT CARD ---------- */

.result-card {
    background: white;
    padding: 28px;
    border-radius: 20px;

    text-align: center;

    border: 1px solid #e5e7eb;

    box-shadow:
        0 8px 25px rgba(0, 0, 0, 0.05);

    margin-top: 20px;
}

.result-emoji {
    font-size: 65px;
}

.result-emotion {
    font-size: 36px;
    font-weight: 800;
    color: #4f46e5;
    margin-top: 5px;
}

.result-confidence {
    font-size: 19px;
    font-weight: 600;
    color: #374151;
    margin-top: 8px;
}


/* ---------- STAT CARD ---------- */

.stat-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    text-align: center;

    border: 1px solid #e5e7eb;

    box-shadow:
        0 5px 15px rgba(0, 0, 0, 0.04);
}

.stat-number {
    font-size: 27px;
    font-weight: 800;
    color: #4f46e5;
}

.stat-label {
    font-size: 13px;
    color: #6b7280;
    margin-top: 5px;
}


/* ---------- FOOTER ---------- */

.footer {
    margin-top: 45px;
    padding: 25px;
    text-align: center;

    border-top: 1px solid #e5e7eb;

    color: #6b7280;
    font-size: 14px;
}

.footer-title {
    font-weight: 700;
    color: #374151;
    font-size: 16px;
}


/* ---------- BUTTON ---------- */

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}


/* ---------- FILE UPLOADER ---------- */

section[data-testid="stFileUploader"] {
    background: white;
    border-radius: 15px;
}


/* ---------- MOBILE ---------- */

@media (max-width: 768px) {

    .hero h1 {
        font-size: 30px;
    }

    .hero p {
        font-size: 15px;
    }

    .section-title {
        font-size: 23px;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL INFORMATION
# ============================================================

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


EMOTION_EMOJIS = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲"
}


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

@st.cache_resource
def load_emotion_model():

    interpreter = tf.lite.Interpreter(
        model_path="emotion_model.tflite"
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


interpreter, input_details, output_details = load_emotion_model()


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


face_detector = load_face_detector()


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_emotion(face):

    # Resize
    face = cv2.resize(
        face,
        (48, 48)
    )

    # Normalize
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

    # Send image to TFLite model
    interpreter.set_tensor(
        input_details[0]["index"],
        face
    )

    # Prediction
    interpreter.invoke()

    predictions = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    # Highest probability
    predicted_index = int(
        np.argmax(predictions)
    )

    emotion = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index] * 100
    )

    return (
        emotion,
        confidence,
        predictions
    )


# ============================================================
# FACE DETECTION + PREDICTION
# ============================================================

def analyze_image(image):

    # Convert PIL image to NumPy
    image_array = np.array(image)

    # Handle RGB images
    if len(image_array.shape) == 3:

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

    else:

        gray = image_array

        image_array = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2RGB
        )

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    result_image = image_array.copy()

    results = []

    for index, (x, y, w, h) in enumerate(faces):

        # Crop face
        face = gray[
            y:y + h,
            x:x + w
        ]

        # Predict emotion
        emotion, confidence, probabilities = (
            predict_emotion(face)
        )

        results.append({
            "face": index + 1,
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": probabilities
        })

        # Draw face rectangle
        cv2.rectangle(
            result_image,
            (x, y),
            (x + w, y + h),
            (79, 70, 229),
            3
        )

        # Label
        label = (
            f"{emotion.upper()} "
            f"{confidence:.1f}%"
        )

        cv2.rectangle(
            result_image,
            (x, y - 35),
            (x + w, y),
            (79, 70, 229),
            -1
        )

        cv2.putText(
            result_image,
            label,
            (x + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    return result_image, results


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_results(results):

    if not results:

        st.warning(
            "⚠️ No face detected. "
            "Please try another image with a clear face."
        )

        return

    st.markdown(
        "### 🎯 Prediction Results"
    )

    for result in results:

        emotion = result["emotion"]

        confidence = result["confidence"]

        emoji = EMOTION_EMOJIS.get(
            emotion,
            "🎭"
        )

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-emoji">
                    {emoji}
                </div>

                <div class="result-emotion">
                    {emotion.upper()}
                </div>

                <div class="result-confidence">
                    Confidence: {confidence:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # Probability chart
        probabilities = result[
            "probabilities"
        ]

        chart_data = {
            CLASS_NAMES[i]: float(
                probabilities[i] * 100
            )
            for i in range(
                len(CLASS_NAMES)
            )
        }

        st.markdown(
            "#### 📊 Emotion Probabilities"
        )

        st.bar_chart(
            chart_data,
            height=280
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">

            <div style="font-size:45px;">
                🎭
            </div>

            <h1>Emotion AI</h1>

            <p>
                Facial Emotion Recognition
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        "### 🧭 Navigation"
    )

    selected_mode = st.radio(
        "Select a prediction mode",
        [
            "🏠 Home",
            "🎥 Real-Time Prediction",
            "📸 Webcam Picture",
            "🖼️ Upload Picture"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        "### 🧠 Model"
    )

    st.write("**Architecture:** CNN")

    st.write("**Input:** 48 × 48 × 1")

    st.write("**Output:** 7 Classes")

    st.write("**Model:** TensorFlow Lite")

    st.divider()

    st.markdown(
        "### 🎭 Emotions"
    )

    st.write(
        "😠 Angry  •  🤢 Disgust"
    )

    st.write(
        "😨 Fear  •  😊 Happy"
    )

    st.write(
        "😐 Neutral  •  😢 Sad"
    )

    st.write(
        "😲 Surprise"
    )


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-icon">
            🎭
        </div>

        <h1>
            Facial Emotion Recognition
        </h1>

        <p>
            Detect and analyze human facial emotions
            using Computer Vision and Deep Learning
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HOME
# ============================================================

if selected_mode == "🏠 Home":

    st.markdown(
        '<div class="section-title">'
        'Welcome to Emotion AI 👋'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">

        <h3>🧠 What is this application?</h3>

        <p>
        This application uses a Convolutional Neural Network
        trained on facial expression images to recognize
        seven different human emotions.
        </p>

        <p>
        The system first detects a human face using
        OpenCV Haar Cascade and then sends the detected
        facial region to the trained CNN model for emotion
        classification.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # Feature cards

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">
                    🎥
                </div>

                <div class="feature-title">
                    Real-Time Detection
                </div>

                <div class="feature-text">
                    Detect emotions directly
                    from your webcam.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">
                    📸
                </div>

                <div class="feature-title">
                    Webcam Picture
                </div>

                <div class="feature-text">
                    Capture a picture and
                    analyze the expression.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">
                    🖼️
                </div>

                <div class="feature-title">
                    Image Upload
                </div>

                <div class="feature-text">
                    Upload an image and
                    detect facial emotions.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    # Statistics

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="stat-card">

                <div class="stat-number">
                    7
                </div>

                <div class="stat-label">
                    Emotions
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="stat-card">

                <div class="stat-number">
                    48×48
                </div>

                <div class="stat-label">
                    Image Input
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="stat-card">

                <div class="stat-number">
                    CNN
                </div>

                <div class="stat-label">
                    Deep Learning
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
            <div class="stat-card">

                <div class="stat-number">
                    AI
                </div>

                <div class="stat-label">
                    Prediction
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# REAL-TIME MODE
# ============================================================

elif selected_mode == "🎥 Real-Time Prediction":

    st.markdown(
        '<div class="section-title">'
        '🎥 Real-Time Emotion Detection'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">

        <b>How it works</b>

        <br><br>

        Webcam → Face Detection → Face Preprocessing
        → CNN → Emotion Prediction

        <br><br>

        Click <b>START</b> below and allow browser
        camera access.

        </div>
        """,
        unsafe_allow_html=True
    )


    class EmotionProcessor(
        VideoProcessorBase
    ):

        def recv(self, frame):

            img = frame.to_ndarray(
                format="bgr24"
            )

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (
                x,
                y,
                w,
                h
            ) in faces:

                face = gray[
                    y:y + h,
                    x:x + w
                ]

                emotion, confidence, _ = (
                    predict_emotion(face)
                )

                # Face box
                cv2.rectangle(
                    img,
                    (x, y),
                    (x + w, y + h),
                    (79, 70, 229),
                    3
                )

                label = (
                    f"{emotion.upper()} "
                    f"{confidence:.1f}%"
                )

                # Label background
                cv2.rectangle(
                    img,
                    (x, y - 35),
                    (x + w, y),
                    (79, 70, 229),
                    -1
                )

                cv2.putText(
                    img,
                    label,
                    (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24"
            )


    webrtc_streamer(
        key="emotion-webcam",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )


# ============================================================
# WEBCAM PICTURE MODE
# ============================================================

elif selected_mode == "📸 Webcam Picture":

    st.markdown(
        '<div class="section-title">'
        '📸 Webcam Picture Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">

        Take a photo using your webcam.
        The application will detect the face and
        predict the emotion.

        </div>
        """,
        unsafe_allow_html=True
    )

    camera_image = st.camera_input(
        "📷 Take a picture"
    )

    if camera_image is not None:

        image = Image.open(
            camera_image
        )

        result_image, results = (
            analyze_image(image)
        )

        st.image(
            result_image,
            caption="Detected Face & Emotion",
            use_container_width=True
        )

        display_results(results)


# ============================================================
# UPLOAD MODE
# ============================================================

elif selected_mode == "🖼️ Upload Picture":

    st.markdown(
        '<div class="section-title">'
        '🖼️ Upload Picture Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">

        Upload a JPG, JPEG or PNG image.
        The application automatically detects faces
        and predicts their emotions.

        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "📁 Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        result_image, results = (
            analyze_image(image)
        )

        st.image(
            result_image,
            caption="Detected Face & Emotion",
            use_container_width=True
        )

        display_results(results)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <div class="footer-title">
            🎭 Facial Emotion Recognition
        </div>

        <br>

        CNN • OpenCV • TensorFlow Lite • Streamlit

        <br><br>

        Computer Vision & Deep Learning Project

    </div>
    """,
    unsafe_allow_html=True
)