                           🛣️ X-PotholeNet Pro  
 Real-Time Multi-Model Pothole Detection with Explainable Severity Analysis



 📌 Introduction

X-PotholeNet Pro is an advanced AI-powered road damage detection and severity analysis system developed for real-time pothole monitoring and road safety assessment.

The system uses a **dual YOLOv8 ensemble architecture**, machine learning-based severity classification, explainable AI techniques, and multi-modal real-time deployment to identify potholes from:

- Uploaded images
- Camera captures
- Live webcam streams
- Uploaded road videos

The framework also calculates road risk scores and provides explainable severity reasoning for every pothole detected.

This project was developed as a research-oriented intelligent transportation and road safety solution.



🎯 Objectives

- Detect potholes in real-time using deep learning
- Classify pothole severity into:
  - Low
  - Medium
  - High
- Reduce false detections using multi-model filtering
- Provide Explainable AI outputs
- Generate road risk assessment
- Support image, webcam, and video processing
- Create a deployable intelligent road monitoring system



 📄 Research Publication

    Published Research Paper

 Title:" X-Potholenet Pro Real-Time Multi-Model Pothole Detection With Explainable Severity Analysis"

📘 Journal: International Journal of Creative and Open Research in Engineering and Management (IJCOPE)  
📅 Published: May 2026  
📑 Volume: 02 Issue 05  
🔖 ISSN: 3108-1754  
📌 DOI: https://doi.org/10.55041/ijcope.v2i5.041


  🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core Development |
| Streamlit | Interactive Dashboard |
| Streamlit-WebRTC | Live Webcam Streaming |
| OpenCV | Image & Video Processing |
| YOLOv8 | Object Detection |
| Ultralytics | YOLO Framework |
| NumPy | Numerical Operations |
| Pandas | Data Processing |
| Matplotlib | Visualization & Analytics |
| Scikit-learn | Severity Classification |
| AV | Video Streaming Backend |



        🧠 AI Architecture

    Triple-Model Detection Pipeline

The system uses three AI detection models simultaneously.

| Model | Purpose |
|---|---|
| YOLOv8n COCO Model | Vehicle & Person Detection |
| Custom YOLOv8 Model 1 | Pothole Detection |
| Custom YOLOv8 Model 2 | Pothole Detection |

COCO Object Detection Model

The COCO model is used for:

- Vehicle detection
- Person detection
- False positive suppression
- Road scene validation

Model File:

```text
yolov8n.pt
```

Custom Pothole Detection Models

Two independent YOLOv8 pothole models are used:

```text
best.pt
pothole_best.pt
```

Purpose:

- Multi-model ensemble pothole detection
- Better confidence calibration
- Improved detection reliability
- Reduced missed detections


  🤖 Severity Classification System

After pothole detection, each pothole undergoes ML-based severity analysis.

   Severity Classes

| Severity |
|---|
| Low |
| Medium |
| High |



   Extracted Features

The framework extracts 7-dimensional severity features.

| Feature | Description |
|---|---|
| area_ratio | Relative pothole size |
| darkness | Depression depth indicator |
| texture | Surface roughness |
| bright_ratio | Water reflection analysis |
| confidence | YOLO confidence |
| model_votes | Ensemble agreement |
| count_context | Nearby pothole density |

   ML Models Used

Three ML classifiers are combined using majority voting.

| Model | Purpose |
|---|---|
| Decision Tree | Rule-based severity classification |
| KNN | Non-linear severity grouping |
| Logistic Regression | Confidence calibration |

Final severity is selected using ensemble voting.

  🧠 Explainable AI (XAI)

The framework generates human-readable explanations such as:

- very large pothole region
- rough damaged texture
- dark depression visible
- water/reflection present
- multi-model agreement

This improves transparency, interpretability, and auditability.



 🚀 System Features

✅ Real-time pothole detection  
✅ Image upload support  
✅ Camera capture mode  
✅ Live webcam processing  
✅ Video upload analysis  
✅ Multi-model YOLO ensemble  
✅ Severity classification  
✅ Explainable AI outputs  
✅ Road risk scoring  
✅ Road safety assessment  
✅ Processed video export  
✅ JSON report generation  
✅ Frame validation system  
✅ False positive suppression  
✅ Video pothole tracking system  

---

# 📂 Project Structure

```text
X-PotholeNet-Pro/
│
├── app.py
├── camera.py
├── detector.py
├── pothole_core.py
├── severity_model.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── best.pt
│   ├── pothole_best.pt
│   └── yolov8n.pt
│
├── paper/
│   ├── X-PotholeNet-Pro-Paper.pdf
│   └── publication-certificate.png
│
└── outputs/
```


## 5 Run Application

```bash
streamlit run app.py
```


# 📸 Application Workflow

## 🖼️ Image Upload

Upload:
- JPG
- PNG
- JPEG
- WEBP

System returns:
- Detected potholes
- Severity labels
- Risk score
- Explainable AI

---

## 📸 Camera Capture

Capture road image directly using camera.

---

## 📹 Live Webcam

Real-time pothole detection using webcam stream.

---

## 🎥 Video Upload

Supported formats:

- MP4
- AVI
- MOV
- MKV

Features:
- Frame processing
- Pothole tracking
- Duplicate suppression
- Annotated output video

---

# 📊 Output Generated

The system provides:

- Bounding box visualization
- Severity-wise pothole count
- Total pothole count
- ML confidence score
- Explainable AI table
- Risk score out of 100
- Road safety status
- Processed video
- Downloadable JSON reports

---

# 📈 Risk Score Formula

```text
Risk Score = min(100,
Low×6 + Medium×20 + High×42 + Total×2)
```

---

# 🚦 Road Status Categories

| Risk Level | Status |
|---|---|
| Safe | Road Looks Safe |
| Low | Minor Road Damage |
| Moderate | Moderate Road Damage |
| High | Drive With Extreme Caution |
| Critical | Unsafe to Drive |

---

# 🔬 Research Contributions

- Dual YOLOv8 ensemble pothole detection
- ML-based severity classification
- Explainable AI framework
- Multi-modal real-time deployment
- Video pothole tracking system
- COCO-based false positive suppression
- Risk score driven road assessment

---

# ⚠️ Challenges Faced

- Dataset preprocessing and conversion
- Multi-model integration
- False positive filtering
- Video frame optimization
- Real-time inference stability
- Webcam streaming issues
- Severity calibration
- Road scene validation

All challenges were resolved during development.

---

 👨‍💻 Authors

- Gangadhar R
- Gururaj Siddayya Hiremath
- Mallikarjunayya S
- M G Nikhil

Computer Science and Engineering  
RV Institute of Technology and Management  
Bengaluru, India - 560078



   📜 License

This project is developed for academic, research, and educational purposes.



 ⭐ Future Enhancements

- GPS pothole mapping
- Edge AI deployment
- Mobile application
- NVIDIA Jetson deployment
- Real-world severity dataset integration
- Cloud analytics dashboard
- Municipal road monitoring integration
