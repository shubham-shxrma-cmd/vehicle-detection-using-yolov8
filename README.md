# Vehicle Detection Using YOLOv8

## Overview

This project is an AI-powered Vehicle Detection System developed using YOLOv8 and Streamlit. It detects vehicles in images, draws bounding boxes around detected objects, and generates a detection report with vehicle classes and confidence scores.

The application provides an easy-to-use web interface where users can upload an image and view the detection results instantly.

---

## Features

- Vehicle detection using YOLOv8
- Streamlit-based web interface
- Detects multiple vehicle types
- Displays bounding boxes on detected vehicles
- Shows detection confidence scores
- Generates a detection report
- Fast and accurate inference

---

## Technologies Used

- Python
- Streamlit
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Pandas

---

## Project Structure

```
Vehicle-Detection-Using-YOLOv8/
│── app.py
│── project.py
│── yolov8s.pt
│── requirements.txt
│── README.md
│── image.jpg
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/shubham-shxrma-cmd/vehicle-detection-using-yolov8.git
```

Go to the project folder:

```bash
cd vehicle-detection-using-yolov8
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

## Sample Output

The application displays:

- Original Image
- Detected Vehicles
- Total Vehicle Count
- Vehicle Class
- Confidence Score

---

## Future Improvements

- Real-time video detection
- Traffic density analysis
- Vehicle counting
- Speed estimation
- License plate recognition

---

## Author

**Shubham Sharma**

GitHub: https://github.com/shubham-shxrma-cmd
