import cv2
import numpy as np
import tensorflow as tf
import base64
import os
import io
from PIL import Image
import re
import argparse

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

# --- Helper Functions ---
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
        print("Model loaded successfully.")
    else:
        raise FileNotFoundError("Model file not found.")
except Exception as e:
    print("Error loading model:", e)
    exit()

# --- Main Analysis Function ---
def analyze_wound_image(input_image_np: np.ndarray, pixels_per_cm2: float = None):
    original_img = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR)
    img_for_model = cv2.resize(original_img, (IMG_WIDTH, IMG_HEIGHT))
    img_array = np.expand_dims(img_for_model, axis=0) / 255.0

    predicted_mask = model.predict(img_array)[0]
    predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(predicted_mask_binary, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    if cv2.countNonZero(mask_resized) == 0:
        return None, "No wound detected."

    current_scale = pixels_per_cm2 if (pixels_per_cm2 and pixels_per_cm2 > 0) else None

    if current_scale is None:
        hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_img, LOWER_BLUE, UPPER_BLUE)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, "Scale missing: Include blue reference or provide pixels_per_cm2."
        ref_contour = max(contours, key=cv2.contourArea)
        ref_area_pixels = cv2.contourArea(ref_contour)
        current_scale = ref_area_pixels / REFERENCE_AREA_CM2

    wound_contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(wound_contours, key=cv2.contourArea)
    area_cm2 = cv2.contourArea(main_contour) / current_scale

    hsv_full = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    tissue_pixel_counts = {t["name"]: 0 for t in TISSUE_TYPES}
    final_overlay = original_img.copy()

    for tissue in TISSUE_TYPES:
        color_mask = cv2.inRange(hsv_full, tissue["lower"], tissue["upper"])
        tissue_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask_resized)
        count = cv2.countNonZero(tissue_mask)
        tissue_pixel_counts[tissue["name"]] = count
        color_overlay = np.zeros_like(original_img)
        color_overlay[tissue_mask > 0] = tissue["color"]
        final_overlay = cv2.addWeighted(final_overlay, 1, color_overlay, 0.6, 0)

    cv2.drawContours(final_overlay, [main_contour], -1, (0, 255, 0), 2)

    total_pixels = sum(tissue_pixel_counts.values())
    tissue_percentages = {n: (c / total_pixels) * 100 for n, c in tissue_pixel_counts.items()}

    return {
        "area_cm2": round(area_cm2, 2),
        "tissue_percentages": {k: round(v, 1) for k, v in tissue_percentages.items() if v > 0.1},
        "clinical_advice": generate_advice(tissue_percentages),
        "overlay_image": final_overlay
    }, None

# --- CLI Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Wound Analysis for NPWT")
    parser.add_argument("--image", type=str, required=True, help="Path to wound image")
    parser.add_argument("--scale", type=float, default=None, help="Pixels per cm^2 (optional)")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)

    result, error = analyze_wound_image(image_np, args.scale)

    if error:
        print("Error:", error)
    else:
        print("\n--- Wound Analysis Report ---")
        print("Wound Area (cm²):", result["area_cm2"])
        print("Tissue Composition (%):")
        for k, v in result["tissue_percentages"].items():
            print(f"  {k}: {v}%")
        print("\nClinical Suggestions:")
        for advice in result["clinical_advice"]:
            print("-", advice)

        output_path = "output_overlay.jpg"
        cv2.imwrite(output_path, result["overlay_image"])
        print(f"\nOverlay saved as: {output_path}")
