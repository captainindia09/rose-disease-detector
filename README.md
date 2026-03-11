---
title: Rose Leaf Disease Detector
emoji: 🌹
colorFrom: red
colorTo: pink
sdk: streamlit
sdk_version: 1.31.1
app_file: app.py
pinned: false
---

# 🌹 AI Rose Leaf Disease Detector

This application uses a custom-trained **YOLOv8 classification model** to detect diseases in rose leaves with over **99% accuracy**.

### 🛠️ Technology Stack:
- **Model**: YOLOv8 (Ultralytics)
- **UI**: Streamlit
- **Deployment**: Hugging Face Spaces
- **Libraries**: PyTorch, OpenCV, Pillow

### 📋 Detected Classes:
- Healthy Rose Leaf
- Rose Rust
- Rose Sawfly (Rose Slug)

### 🚀 How to use locally:
1. Clone the repo
2. Run `pip install -r requirements.txt`
3. Run `streamlit run app.py`

Created for GenAI Agricultural Hackathon.
