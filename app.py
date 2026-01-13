import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
import base64
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import io
from PIL import Image
import re

# --- Part 1: Constants and Configuration ---
MODEL_PATH = 'wound_segmentation_model.h5'
IMG_HEIGHT = 256
IMG_WIDTH = 256
REFERENCE_AREA_CM2 = 4.0
LOWER_BLUE = np.array([95, 60, 100])
UPPER_BLUE = np.array([120, 255, 255])

TISSUE_TYPES = [
    {"name": "Black Necrotic", "id": 1, "color": (0, 0, 0), "lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 50])},
    {"name": "Brown Necrotic", "id": 2, "color": (19, 69, 139), "lower": np.array([10, 50, 50]), "upper": np.array([20, 150, 100])},
    {"name": "Yellow Slough", "id": 3, "color": (0, 255, 255), "lower": np.array([18, 60, 70]), "upper": np.array([40, 255, 255])},
    {"name": "White Fibrin", "id": 4, "color": (230, 230, 230), "lower": np.array([0, 0, 130]), "upper": np.array([180, 50, 255])},
    {"name": "Unhealthy", "id": 5, "color": (0, 0, 139), "lower": np.array([0, 100, 50]), "upper": np.array([10, 255, 150])},
    {"name": "Granulation", "id": 6, "color": (0, 0, 255), "lower": np.array([165, 100, 80]), "upper": np.array([180, 255, 255])},
    {"name": "Epithelializing", "id": 7, "color": (203, 192, 255), "lower": np.array([145, 15, 130]), "upper": np.array([170, 120, 255])}
]

# --- Part 2: Helper Functions ---
def iou(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def generate_advice(percentages):
    advice = []
    slough = percentages.get("Yellow Slough", 0) + percentages.get("White Fibrin", 0)
    necrotic = percentages.get("Black Necrotic", 0) + percentages.get("Brown Necrotic", 0)
    granulation = percentages.get("Granulation", 0)
    epithelializing = percentages.get("Epithelializing", 0)
    
    if necrotic > 10: advice.append("Significant necrotic tissue present. Consider debridement.")
    if slough > 20: advice.append("Slough/fibrin observed. Consider autolytic debridement dressings.")
    if granulation > 70 and not epithelializing > 10: advice.append("Good granulation. Maintain moist environment.")
    if epithelializing > 10: advice.append("Wound is actively epithelializing.")
    if not advice: advice.append("Continue standard wound care and monitor.")
    return advice

# Load Model
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'iou': iou})
        print("--- Model loaded successfully ---")
    else:
        model = None
        print("--- Model file not found ---")
except Exception as e:
    model = None
    print(f"--- Error loading model: {e} ---")

# --- Part 3: Main Analysis Logic (Reframing Enabled) ---
def analyze_wound_image(input_image_np: np.ndarray, pixels_per_cm2: float = None):
    if model is None: 
        return {"status": "error", "message": "Model not available."}
    
    # Preprocessing
    original_img = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR)
    img_for_model = cv2.resize(original_img, (IMG_WIDTH, IMG_HEIGHT))
    img_array = np.expand_dims(img_for_model, axis=0) / 255.0
    
    # Model Prediction
    predicted_mask = model.predict(img_array)[0]
    predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(predicted_mask_binary, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    if cv2.countNonZero(mask_resized) == 0:
        return {"status": "error", "message": "No wound detected in the image."}

    # --- REFRAMING LOGIC: Scale Determination ---
    # We check if pixels_per_cm2 is valid (not None and not 0)
    current_scale = pixels_per_cm2 if (pixels_per_cm2 and pixels_per_cm2 > 0) else None

    if current_scale is None:
        # Search for blue reference object
        hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_img, LOWER_BLUE, UPPER_BLUE)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"status": "error", "message": "Scale missing: Please include blue reference or provide pixels_per_cm2."}

        ref_contour = max(contours, key=cv2.contourArea)
        ref_area_pixels = cv2.contourArea(ref_contour)
        
        if ref_area_pixels <= 0:
            return {"status": "error", "message": "Reference object found but area is zero."}
            
        current_scale = ref_area_pixels / REFERENCE_AREA_CM2
    
    # Calculate Area
    wound_contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(wound_contours, key=cv2.contourArea) if wound_contours else None
    area_cm2 = (cv2.contourArea(main_contour) / current_scale) if main_contour is not None else 0

    # Tissue Analysis
    hsv_full = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    tissue_pixel_counts = {t["name"]: 0 for t in TISSUE_TYPES}
    final_overlay = original_img.copy()
    
    for tissue in TISSUE_TYPES:
        color_mask = cv2.inRange(hsv_full, tissue["lower"], tissue["upper"])
        tissue_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask_resized)
        count = cv2.countNonZero(tissue_mask)
        tissue_pixel_counts[tissue["name"]] = count
        
        # Add color to overlay
        color_overlay = np.zeros_like(original_img)
        color_overlay[tissue_mask > 0] = tissue["color"]
        final_overlay = cv2.addWeighted(final_overlay, 1, color_overlay, 0.6, 0)

    if main_contour is not None:
        cv2.drawContours(final_overlay, [main_contour], -1, (0, 255, 0), 2)

    # Metrics
    total_tissue_pixels = sum(tissue_pixel_counts.values())
    tissue_percentages = {n: (c / total_tissue_pixels) * 100 for n, c in tissue_pixel_counts.items()} if total_tissue_pixels > 0 else {}

    # Encode Image
    _, buffer = cv2.imencode('.jpg', final_overlay)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "status": "success",
        "analysis": {
            "total_area_cm2": round(area_cm2, 2),
            "pixels_per_cm2": round(float(current_scale), 4), # Returned for future reframing calls
            "tissue_composition": {name: round(p, 1) for name, p in tissue_percentages.items() if p > 0.1}
        },
        "clinical_suggestions": generate_advice(tissue_percentages),
        "processed_image_base64": f"data:image/jpeg;base64,{img_b64}"
    }

# --- Part 4: API Endpoints ---
app = FastAPI()

class AnalyzeRequest(BaseModel):
    image_base64: str
    pixels_per_cm2: float = None # Can be omitted, null, or a value

@app.post("/analyze")
def analyze_api_endpoint(request: AnalyzeRequest):
    try:
        # Decode Image
        base64_str = request.image_base64
        if "," in base64_str:
            base64_str = re.sub(r'^data:image/.+;base64,', '', base64_str)

        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image.convert('RGB'))
        
        # Process with optional pixels_per_cm2
        result = analyze_wound_image(image_np, pixels_per_cm2=request.pixels_per_cm2)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result["message"])
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Gradio Setup (for manual UI testing)
def gradio_fn(img):
    if img is None: return None
    return analyze_wound_image(img)

demo = gr.Interface(fn=gradio_fn, inputs=gr.Image(type="numpy"), outputs=gr.JSON())
app = gr.mount_gradio_app(app, demo, path="/")