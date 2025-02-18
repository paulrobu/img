import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# Load YOLO model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Target width
TARGET_WIDTH = 1800

def crop_opencv(image):
    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cropped = image.crop((x, y, x + w, y + h))
        return cropped.resize((TARGET_WIDTH, int(TARGET_WIDTH * h / w)))
    
    return image

def crop_yolo(image):
    results = yolo_model(image)
    boxes = results.xyxy[0].numpy()
    
    if len(boxes) > 0:
        x1, y1, x2, y2, _, _ = boxes[0]
        cropped = image.crop((int(x1), int(y1), int(x2), int(y2)))
        return cropped.resize((TARGET_WIDTH, int(TARGET_WIDTH * (y2 - y1) / (x2 - x1))))
    
    return image

def crop_golden_spiral(image):
    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, _, _, maxLoc = cv2.minMaxLoc(blurred)
    
    img_width, img_height = image.size
    golden_x = int(img_width * 0.618)
    golden_y = int(img_height * 0.618)
    
    new_x1 = max(0, golden_x - img_width // 3)
    new_y1 = max(0, golden_y - img_height // 3)
    new_x2 = min(img_width, new_x1 + img_width // 1.5)
    new_y2 = min(img_height, new_y1 + img_height // 1.5)
    
    cropped = image.crop((new_x1, new_y1, new_x2, new_y2))
    return cropped.resize((TARGET_WIDTH, int(TARGET_WIDTH * (new_y2 - new_y1) / (new_x2 - new_x1))))

def save_image(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=100, optimize=True)
    return buffer.getvalue()

# Streamlit UI
st.title("Procesare Automată a Imaginilor")
uploaded_file = st.file_uploader("Încarcă o imagine", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagine Originală", use_column_width=True)
    
    if st.button("Procesează Imaginea"):
        opencv_cropped = crop_opencv(image)
        yolo_cropped = crop_yolo(image)
        golden_cropped = crop_golden_spiral(image)
        
        st.image(opencv_cropped, caption="OpenCV Crop", use_column_width=True)
        st.image(yolo_cropped, caption="YOLO Crop", use_column_width=True)
        st.image(golden_cropped, caption="Golden Spiral Crop", use_column_width=True)
        
        st.download_button("Descarcă OpenCV", data=save_image(opencv_cropped), file_name="opencv_crop.jpg", mime="image/jpeg")
        st.download_button("Descarcă YOLO", data=save_image(yolo_cropped), file_name="yolo_crop.jpg", mime="image/jpeg")
        st.download_button("Descarcă Golden Spiral", data=save_image(golden_cropped), file_name="golden_spiral_crop.jpg", mime="image/jpeg")
