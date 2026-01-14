# Automated Wound Analysis for NPWT using Deep Learning

## Overview

This project presents an AI-driven Computer Vision and Clinical Decision Support system for **Automated Wound Segmentation, Tissue Classification, and Area Estimation** to assist **Negative Pressure Wound Therapy (NPWT)** planning and monitoring.

The system combines:

- **U-Net based deep learning segmentation** trained on **3000+ annotated wound images**
- **Color-based tissue clustering (K-means, k=10)** with adaptive cluster merging
- **Scale-aware wound area computation** using reference calibration
- **LLM-powered NPWT clinical advisory** using RAG over medical literature
- **FastAPI backend with Gradio interface** for real-time clinical interaction

Achieved performance:
- **98–99% segmentation accuracy**
- **97%+ wound area measurement accuracy**
- Robust operation across varying lighting, skin tones, and wound morphologies

---

## Key Features

### 1. Deep Learning Wound Segmentation
- Architecture: **U-Net (CNN)**
- Training data: **3000+ pixel-wise annotated chronic wound images**
- Output: Multi-class tissue masks
  - Granulation  
  - Slough (yellow / white fibrin)  
  - Necrotic (black / brown)  
  - Epithelializing  
- Performance: **98–99% segmentation accuracy**

### 2. Tissue Composition Analysis
- Post-processing using **K-means clustering (k=10) in HSV color space**
- Adaptive **duplicate-cluster merging** for illumination invariance
- Computes percentage distribution of each tissue class for clinical staging

### 3. Area & Scale Estimation
- Scale calibration using:
  - **Blue reference marker (known area = 4 cm²)**, or
  - User-provided pixels-per-cm² (longitudinal follow-up)
- Achieved **97%+ accuracy** in wound area estimation compared to ground truth measurements

### 4. NPWT Clinical Decision Support (LLM + RAG)
- Built using **LangChain + Gemini API**
- Knowledge base:
  - **16 NPWT textbooks and peer-reviewed research papers**
- Provides:
  - Therapy suitability assessment
  - Pressure range suggestions
  - Dressing change frequency
  - Risk warnings for necrosis, infection, and poor granulation
- Retrieval-Augmented Generation (RAG) ensures evidence-grounded responses

### 5. API & Web Interface
- Backend: **FastAPI microservice**
- Frontend: **Gradio interactive UI**
- Supports:
  - Base64 image input (hospital system integration)
  - Real-time inference
  - Overlay visualization (mask, contours, tissue maps)
  - Edge-ready deployment

---

## Architecture

- **Segmentation Network:** U-Net (TensorFlow)
- **Color Analysis:** K-means clustering + morphological refinement
- **Scale Estimation:** Reference-based pixel-to-cm² calibration
- **Clinical Reasoning:** Rule layer + LLM (Gemini) with RAG
- **Deployment:** FastAPI + Gradio (REST + UI)

---

## Online Demo (Hugging Face Spaces)

🔗 https://huggingface.co/spaces/Arshadpk/wound-Analysis

The demo supports:
- Uploading wound images
- Pixel-wise segmentation (98–99% accuracy)
- Tissue composition breakdown
- Reference-based area measurement (97%+ accuracy)
- Evidence-grounded NPWT treatment recommendations

---

## Sample Test Images

| File | Description |
|------|-------------|
| `test_wound.jpg` | Chronic wound image for segmentation & tissue analysis |
| `reference_marker.jpg` | Blue calibration marker (area = 4 cm²) |

Usage:
- Place the reference marker near the wound during imaging for automatic scale recovery.
- Alternatively, supply `pixels_per_cm2` through the API for follow-up assessments.

---

## Clinical Impact

- Objective wound staging and monitoring
- Accurate area tracking for NPWT pressure planning
- Evidence-grounded therapy recommendations
- Reduced inter-observer variability
- Supports edge deployment in resource-limited clinical settings
