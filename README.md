# Automated Wound Analysis for NPWT using Deep Learning

## Overview

This project presents an AI-driven Computer Vision system for **Automated Wound Segmentation and Tissue Classification** to support **Negative Pressure Wound Therapy (NPWT)** clinical decision-making.

The system integrates:
- Deep Learning–based wound segmentation (U-Net style CNN)
- Tissue-type classification using HSV-based color modeling
- Real-world area estimation using reference scaling
- Clinical advisory generation
- FastAPI backend with Gradio interface for real-time usage

---

## Key Features

### 1. Wound Segmentation
- Deep CNN model (`wound_segmentation_model.h5`)
- Pixel-wise wound boundary detection
- Robust to lighting, angle, and scale variations

### 2. Tissue Composition Analysis
Automatically identifies:
- Granulation tissue  
- Slough (yellow / white fibrin)  
- Necrotic tissue (black / brown)  
- Epithelializing regions  

Percentage distribution is computed for clinical assessment.

### 3. Area & Scale Estimation
- Uses either:
  - Blue reference marker (automatic scale detection), or
  - User-provided pixels-per-cm² (for longitudinal tracking)
- Computes wound area in cm².

### 4. Clinical Decision Support
Generates treatment suggestions for NPWT based on:
- Necrotic burden
- Slough percentage
- Granulation dominance
- Epithelialization stage

### 5. API + Web Interface
- REST API using FastAPI
- Interactive UI using Gradio
- Base64 image input support for hospital system integration

---

## Architecture

- CNN Segmentation Network (TensorFlow)
- Post-processing with OpenCV
- Graphical overlay and contour extraction
- Clinical rule-based reasoning layer
- FastAPI microservice + Gradio frontend

---

## Online Demo (Hugging Face Spaces)

Try the live model here:

🔗 https://huggingface.co/spaces/Arshadpk/wound-Analysis

The demo allows:
- Uploading wound images
- Automatic segmentation
- Tissue composition analysis
- Wound area measurement using blue reference object
- Clinical NPWT decision support

---

## Sample Test Images

Example images are provided in the repo:

| File | Description |
|------|-------------|
| `test_wound.jpg` | Sample chronic wound image for segmentation and tissue analysis |
| `reference_marker.jpg` | Blue reference object (known area = 4 cm²) for scale calibration |

Usage:
- Place the blue marker near the wound when capturing images.
- Or provide `pixels_per_cm2` manually via API for longitudinal follow-up.


