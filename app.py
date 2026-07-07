import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import tempfile
from PIL import Image

# ==============================
# Load YOLOv8 Model
# ==============================
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="🚗 Vehicle Detection", layout="wide")
st.title("🚗 Vehicle Detection & Counting using YOLOv8")

st.markdown("Upload an *image* to detect and count vehicles.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        img_path = temp.name

    # Run YOLO prediction
    results = model.predict(img_path, conf=0.3)

    # Vehicle classes
vehicle_classes = ['car', 'bus', 'truck', 'motorbike']

# Extract detections manually
detections = []
for box in results[0].boxes:
    cls_id = int(box.cls[0])  # class id
    cls_name = results[0].names[cls_id]  # class name
    conf = float(box.conf[0])  # confidence
    if cls_name in vehicle_classes:
        detections.append({"name": cls_name, "confidence": conf})

import pandas as pd
vehicles = pd.DataFrame(detections)

# Show annotated result
st.image(results[0].plot(), caption="Detected Vehicles", use_column_width=True)

# Report
st.subheader("📊 Detection Report")
st.write(f"*Total Vehicles Detected:* {len(vehicles)}")

if not vehicles.empty:
    st.dataframe(vehicles)
else:
    st.warning("⚠ No vehicles detected!")