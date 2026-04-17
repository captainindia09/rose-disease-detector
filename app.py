import torch
# Fix for Streamlit file watcher conflict with PyTorch
torch.classes.__path__ = [] 

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Rose Health AI",
    page_icon="🌹",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    /* Main Background - Clean and High Contrast */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Force Dark Text for ALL elements */
    h1, h2, h3, h4, h5, h6, p, span, li, label, .stMarkdown {
        color: #0f172a !important; /* Extremely Dark Blue/Black */
    }
    
    .subtitle {
        color: #334155 !important;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Result Card - Dark background with white text for maximum pop */
    .result-card {
        padding: 25px;
        border-radius: 15px;
        background-color: #1e293b !important; /* Dark Slate */
        border-left: 8px solid #10b981;
        margin-top: 25px;
    }

    .result-card h1, .result-card h2, .result-card h3, .result-card h4, .result-card p, .result-card li {
        color: #ffffff !important; /* White text on dark card */
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        background-color: #dc2626;
        color: white !important;
        font-weight: 700;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# --- APP HEADER ---
st.markdown('<h1 style="text-align: center;">🌹 AI Rose Leaf Disease Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle" style="text-align: center;">Upload a leaf photo to diagnose health issues and receive professional treatment protocols instantly.</p>', unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = "best_v2.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.error(f"Model file '{model_path}' not found. Please ensure it is in the app directory.")
        return None

model = load_model()

# --- TREATMENT DATABASE ---
TREATMENTS = {
    "Healthy_Leaf_Rose": {
        "status": "Healthy & Thriving! ✨",
        "advice": [
            "Maintain current watering schedule (check soil moisture).",
            "Ensure the plant gets 6-8 hours of sunlight.",
            "Apply balanced fertilizer once a month.",
            "Regularly prune old or yellowing leaves to encourage growth."
        ],
        "severity": "Low"
    },
    "Rose_Rust": {
        "status": "Infection Detected: Rose Rust 🍂",
        "advice": [
            "Immediately remove and destroy infected leaves (do not compost them).",
            "Avoid overhead watering; water the base of the plant to keep leaves dry.",
            "Apply a sulfur-based fungicide or Neem oil regularly.",
            "Increase air circulation by thinning out inner branches."
        ],
        "severity": "Medium"
    },
    "Rose_sawfly_Rose_slug": {
        "status": "Pest Attack Detected: Sawfly/Rose Slug 🐛",
        "advice": [
            "Pick off the green caterpillars (larvae) by hand if the infestation is small.",
            "Spray the undersides of leaves with a strong stream of water to dislodge them.",
            "Apply Insecticidal Soap or Neem oil to the foliage.",
            "Keep the area around the base clean of debris where larvae might hide."
        ],
        "severity": "High"
    }
}

# --- IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Choose a rose leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Prediction Button
    if st.button("🔍 Diagnose Heart Health"):
        with st.spinner('Analyzing plant health...'):
            if model:
                # Run Inference
                results = model(image)
                
                # Process Results
                res = results[0]
                names = res.names
                probs = res.probs
                
                top1_idx = probs.top1
                confidence = float(probs.top1conf)
                predicted_class = names[top1_idx]
                
                # UI Layout for Results
                st.markdown(f"### Diagnosis: **{predicted_class.replace('_', ' ')}**")
                st.progress(confidence)
                st.write(f"Confidence: **{confidence:.2%}**")

                # Treatment Recommendations
                info = TREATMENTS.get(predicted_class, {})
                
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.subheader(info.get("status", "Unknown Condition"))
                
                st.markdown("#### Recommended Actions:")
                for step in info.get("advice", []):
                    st.markdown(f"- {step}")
                
                st.markdown("</div>", unsafe_allow_html=True)

                # Top 3 Probabilities
                with st.expander("View detailed probabilities"):
                    top5_indices = probs.top5
                    top5_confs = probs.top5conf
                    for i, idx in enumerate(top5_indices):
                        class_name = names[idx].replace('_', ' ')
                        conf = float(top5_confs[i])
                        st.write(f"{class_name}: {conf:.2%}")
                        st.progress(conf)

# --- FOOTER ---
st.divider()
st.markdown("Created for GenAI Agricultural Hackathon • Powered by YOLOv8 Transfer Learning")
