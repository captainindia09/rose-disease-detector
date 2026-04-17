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

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rose-disease-detector-apeiasgzzs9tjwa2w9kly2.streamlit.app/)

**Live Deployment:** [Rose Disease Detector on Streamlit Cloud](https://rose-disease-detector-apeiasgzzs9tjwa2w9kly2.streamlit.app/)

This application is an end-to-end personal Machine Learning project. It leverages a custom-trained **YOLOv8 classification model** to detect and diagnose diseases in rose leaves with high accuracy, instantly providing professional agricultural treatment protocols.

---

## 🛠️ Technology Stack
- **Deep Learning Framework**: PyTorch, Ultralytics YOLOv8
- **Data Engineering**: Python, split-folders
- **Web Application UI**: Streamlit 
- **Deployment & Hosting**: Streamlit Community Cloud
- **Libraries**: OpenCV (Headless), Pillow, NumPy

---

## 📋 Detected Classes
Our model was carefully trained to classify images into one of the following categories:
- **Healthy Rose Leaf** ✨
- **Rose Rust** 🍂
- **Rose Sawfly (Rose Slug)** 🐛

---

## 🏗️ How I Built This From Scratch (The Pipeline)

### Phase 1: Data Consolidation & Preprocessing
The dataset started as a collection of unstructured raw images. I built `consolidate_data.py` to:
1. Programmatically organize the images into unified class folders.
2. Filter out corrupted or incompatible image formats.
3. Use the `split-folders` library to cleanly divide the dataset into `train/`, `val/`, and `test/` splits exactly matched to YOLOv8's expected hierarchical structure (`dataset_v2`).

### Phase 2: Model Training (YOLOv8 Transfer Learning)
Instead of building a CNN from scratch, I utilized transfer learning using the state-of-the-art **YOLOv8 image classification network** (`yolov8n-cls.pt`). 
- Using `train_and_evaluate.py` and `rose_disease_prediction.ipynb`, I initialized the pre-trained weights and fine-tuned the model entirely on the custom rose leaf dataset. 
- The model trained for multiple epochs until it converged, analyzing thousands of epochs worth of image augmentations.
- The absolute best weights were saved as `best_v2.pt`, shrinking a multi-gigabyte training process into a highly efficient `< 3MB` model file ready for inference.

### Phase 3: Web Application Development
I built `app.py` using **Streamlit** to wrap the PyTorch model in an intuitive User Interface. Features include:
- A drag-and-drop Image Uploader.
- Instant model inference via `@st.cache_resource` for real-time latency.
- Dynamic **Treatment Protocol Database**: Based on what the AI diagnoses, the app actively renders a customized remediation checklist (e.g., specific fungicides, soil tips, pest control methods).
- Custom UI/UX: Enforced through native Streamlit `.streamlit/config.toml` global variables to provide a clean, high-contrast Light Theme experience.

### Phase 4: Production Deployment
To take the app from my local machine to the internet:
- Switched to `opencv-python-headless` in `requirements.txt` to strip out unnecessary OS display dependencies and keep cloud builds fast.
- Defined specific runtime controls (`runtime.txt` locked to Python 3.11).
- Connected the GitHub repository to **Streamlit Community Cloud** to create an automated CI/CD pipeline. Every new commit pushed to the repository automatically deploys to the live web address.

---

## 🚀 How to Run Locally

If you'd like to test or modify the code on your own machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/captainindia09/rose-disease-detector.git
   cd rose-disease-detector
   ```

2. **Install the dependencies:**
   *(Ensure you have Python 3.11+ installed)*
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit server:**
   ```bash
   streamlit run app.py
   ```

4. **Navigate to Localhost:** 
   Open your browser and visit `http://localhost:8501`.

---
*Created with ❤️*
