import tarfile
import urllib.request
import os

import numpy as np
import streamlit as st
import cv2

@st.cache(ttl=24*3600)
def dowload_model():
    DATA_DIR = os.path.join(os.getcwd(), 'data')
    MODELS_DIR = os.path.join(DATA_DIR, 'yolo')
    for dir in [DATA_DIR, MODELS_DIR]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    # Download and extract model
    MODEL_NAME = 'yolov3'
    MODEL_WEIGHTS_FILENAME = MODEL_NAME + '.weights'
    MODEL_CONFIG_FILENAME = MODEL_NAME + '.cfg'
    MODELS_DOWNLOAD_BASE = 'https://pjreddie.com/media/files/'
    MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_WEIGHTS_FILENAME
    PATH_TO_WEIGHTS = os.path.join(MODELS_DIR, MODEL_WEIGHTS_FILENAME)
    PATH_TO_CONFIG = os.path.join(MODELS_DIR, MODEL_CONFIG_FILENAME)
    if not os.path.exists(PATH_TO_WEIGHTS):
        print('Downloading model. This may take a while... ', end='')
        st.write('Downloading model. This may take a while... ')
        urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_WEIGHTS)
        print('Done')
        st.write('Download completed ... ')


def load_model():
    # Load YOLOv3 config and weights
    model = cv2.dnn.readNet("data/yolo/yolov3.weights", "data/yolo/yolov3.cfg")
    return model


def load_labels():
    # Get class labels
    with open("data/yolo/coco.names", "r") as f:
        labels = [line.strip() for line in f.readlines()]

    return labels


dowload_model()
net = load_model()

# Get class labels
classes = load_labels()

# Set minimum confidence threshold
conf_threshold = 0.5

# Define Streamlit app
st.title("Object Detection with YOLOv3")

# Define input image upload
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

# Check if image has been uploaded
if uploaded_file is not None:
    # Read image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Get image dimensions
    height, width, channels = img.shape

    # Create blob from image
    blob = cv2.dnn.blobFromImage(
        img, 1/255, (416, 416), swapRB=True, crop=False)

    # Set input to neural network
    net.setInput(blob)

    # Get output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Forward pass through neural network
    outputs = net.forward(output_layer_names)

    # Postprocess outputs
    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Draw boxes and labels on image
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, conf_threshold, 0.4).flatten()
    for i in indices:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]
        color = (255, 0, 0)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, f"{label}: {confidence:.2f}",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show output image
    st.image(img, channels="BGR")
