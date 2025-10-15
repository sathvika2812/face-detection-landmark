import streamlit as st
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np

# ----------------------------
# Initialize Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="Face Detection & Landmarks", layout="wide")
st.title("Clickable Face Detection with Landmarks")

# ----------------------------
# Input type selection
input_type = st.radio("Select Input Type:", ["Upload Image", "Webcam"])

# ----------------------------
def detect_faces(image):
    """Detect faces in the image and return bounding boxes"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = []
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)
        if results.detections:
            ih, iw, _ = image.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                faces.append(bbox)
    return faces

def draw_landmarks(image):
    """Draw facial landmarks"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                )
    return image

# ----------------------------
# IMAGE UPLOAD
if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces = detect_faces(image)

        # Draw bounding boxes
        for bbox in faces:
            x, y, w, h = bbox
            cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)

        # Draw landmarks
        image = draw_landmarks(image)
        
        # Display image
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Detected {len(faces)} face(s)", use_column_width=True)

# ----------------------------
# WEBCAM
else:
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break

        faces = detect_faces(frame)
        for bbox in faces:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        frame = draw_landmarks(frame)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
