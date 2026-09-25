# 😊 Facial Emotion Recognition

### CNN-Based Facial Emotion Recognition with Real-Time and Image Prediction

A deep learning application that detects human faces and predicts facial emotions using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset.

The project provides three different prediction modes through a Streamlit web application:

- 🎥 Real-Time Webcam Prediction
- 📸 Webcam Picture Prediction
- 🖼️ Upload Picture Prediction

---

## 🚀 Live Demo

👉 **[Open Facial Emotion Recognition App]([https://customer-churn-prediction-nwsewnvcutao5taq8kv8th.streamlit.app/](https://facialemotionrecognition-ukjkc87hejsthbnc7pzrhf.streamlit.app/))**

---

## 📌 Project Overview

Facial expressions provide useful information about a person's emotional state. This project uses computer vision and deep learning to automatically recognize facial expressions from images and webcam frames.

The system first detects the face using OpenCV's Haar Cascade classifier. The detected face is converted into a grayscale 48×48 image and normalized before being passed to a trained CNN model.

The CNN then predicts one of seven emotions.

### Supported Emotions

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

---

## ✨ Features

### 🎥 1. Real-Time Prediction

Uses the device webcam to continuously detect faces and predict emotions in real time.

The application displays:

- Face bounding box
- Predicted emotion
- Prediction confidence

---

### 📸 2. Webcam Picture Prediction

Allows the user to capture a single picture using the webcam.

The application then:

1. Detects the face
2. Preprocesses the face
3. Predicts the emotion
4. Displays the result

---

### 🖼️ 3. Upload Picture Prediction

Users can upload:

- JPG
- JPEG
- PNG

The application detects faces in the uploaded image and predicts the emotion of each detected face.

---

## 🧠 Machine Learning Workflow

```text
FER-2013 Dataset
       ↓
Data Exploration
       ↓
Image Preprocessing
       ↓
Grayscale Images
       ↓
48 × 48 Image Resizing
       ↓
Pixel Normalization
       ↓
CNN Model
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Saved Keras Model
       ↓
OpenCV Face Detection
       ↓
Emotion Prediction
       ↓
Streamlit Application
```

---

## 📊 Dataset

The project uses the **FER-2013 (Facial Expression Recognition 2013)** dataset.

The dataset contains grayscale facial images of size:

```text
48 × 48 pixels
```

The model recognizes seven emotion classes:

```text
Angry
Disgust
Fear
Happy
Neutral
Sad
Surprise
```

The dataset is used locally for training and is not included in this repository because of its size and licensing considerations.

---

## 🧠 CNN Architecture

The project uses a Convolutional Neural Network designed for facial emotion classification.

### Architecture

```text
Input
48 × 48 × 1
     ↓
Conv2D
32 Filters
     ↓
MaxPooling
     ↓
Conv2D
64 Filters
     ↓
MaxPooling
     ↓
Conv2D
128 Filters
     ↓
MaxPooling
     ↓
Flatten
     ↓
Dense
128 Neurons
     ↓
Dropout
0.5
     ↓
Output
7 Classes
```

### Why CNN?

CNNs are well suited for image classification because convolutional layers can automatically learn visual patterns such as:

- Edges
- Shapes
- Facial features
- Eyes
- Mouth
- Eyebrows
- Other expression-related patterns

---

## ⚙️ Model Training

The model was compiled using:

- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Metric:** Accuracy

Training also uses:

- Early Stopping
- Learning Rate Reduction

These techniques help control overfitting and improve training efficiency.

---

## 🔍 Image Preprocessing

Before prediction, every detected face goes through the following preprocessing:

```text
Original Face
     ↓
Convert to Grayscale
     ↓
Resize to 48 × 48
     ↓
Normalize Pixel Values
     ↓
Add Channel Dimension
     ↓
CNN Prediction
```

Pixel values are normalized from:

```text
0–255
```

to:

```text
0–1
```

using:

```python
face = face.astype(np.float32) / 255.0
```

---

## 👁️ Face Detection

OpenCV's Haar Cascade classifier is used to detect faces before sending them to the CNN.

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)
```

This separates the face detection task from the emotion classification task.

---

## 🖥️ Application Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | CNN model development |
| OpenCV | Face detection and image processing |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Matplotlib | Data visualization |
| Scikit-learn | Model evaluation |
| Streamlit | Web application |
| Streamlit-WebRTC | Real-time webcam streaming |
| Jupyter Notebook | Model development and experimentation |

---

## 📁 Project Structure

```text
Facial_Emotion_Recognition/
│
├── app.py
├── emotion_model.keras
├── emotion_recognition.ipynb
├── requirements.txt
├── .gitignore
├── README.md
│
├── dataset/
│   ├── train/
│   └── test/
│
└── venv/
```

> `dataset/` and `venv/` are excluded from GitHub using `.gitignore`.

---

## 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Facial_Emotion_Recognition.git
```

### 2. Navigate to the project

```bash
cd Facial_Emotion_Recognition
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

#### Windows

```powershell
.\venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application Locally

Run:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

## 🎥 Using Real-Time Prediction

1. Open the Streamlit application.
2. Select **Real-Time Prediction**.
3. Click **START**.
4. Allow camera access.
5. Position your face in front of the camera.
6. The application detects the face and predicts the emotion.

---

## 📸 Using Webcam Picture Prediction

1. Select **Webcam Picture Prediction**.
2. Allow camera access.
3. Capture a picture.
4. The application detects the face.
5. The predicted emotion and confidence are displayed.

---

## 🖼️ Using Upload Picture Prediction

1. Select **Upload Picture Prediction**.
2. Upload a JPG, JPEG, or PNG image.
3. The application detects faces.
4. Each detected face is classified.
5. The prediction is displayed on the image.

---

## 📈 Model Evaluation

The model was evaluated using:

- Accuracy
- Classification Report
- Confusion Matrix
- Training/Validation Loss
- Training/Validation Accuracy

The evaluation was performed during the model development process in:

```text
emotion_recognition.ipynb
```

---

## 🎯 Project Objectives

The main objectives of this project are:

- Build an image classification model using CNN.
- Understand facial image preprocessing.
- Detect faces using OpenCV.
- Classify facial expressions into seven categories.
- Implement real-time emotion prediction.
- Build a user-friendly Streamlit application.
- Deploy the application as a web-based ML project.

---

## 🔮 Future Improvements

Possible future improvements include:

- Improve model accuracy using a deeper CNN architecture.
- Use data augmentation to improve generalization.
- Experiment with transfer learning.
- Improve real-time prediction performance.
- Add prediction history.
- Add emotion statistics and visualization.
- Improve face detection under different lighting conditions.
- Explore more advanced face detection methods.

---

## ⚠️ Limitations

The predicted emotion represents a **model classification of facial expression**, not a definitive measurement of a person's actual emotional or mental state.

Prediction performance can be affected by:

- Lighting conditions
- Face angle
- Image quality
- Occlusion
- Multiple faces
- Facial expression ambiguity
- Dataset limitations

---

## 📚 Key Concepts Demonstrated

This project demonstrates practical knowledge of:

- Python
- Computer Vision
- Image Processing
- Convolutional Neural Networks
- Deep Learning
- Image Classification
- Model Evaluation
- OpenCV
- TensorFlow/Keras
- Streamlit
- Real-Time Video Processing
- Web Application Deployment

---

## 👨‍💻 Author

**Akash Poojary**

BSc Data Science
