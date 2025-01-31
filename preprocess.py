import cv2
import numpy as np
import os

# Load Pretrained Model (EDSR for Super-Resolution)
MODEL_PATH = "models/EDSR_x3.pb"  # Ensure the model is available
UPSCALE_FACTOR = 3  # Default upscaling factor

# Initialize Super-Resolution Model
sr = cv2.dnn_superres.DnnSuperResImpl_create()

if os.path.exists(MODEL_PATH):
    sr.readModel(MODEL_PATH)
    sr.setModel("edsr", UPSCALE_FACTOR)  # EDSR (Enhanced Deep Residual Networks)
    print(f"[INFO] Super-Resolution model loaded: {MODEL_PATH}")
else:
    print(f"[ERROR] Model not found at {MODEL_PATH}. Super-resolution will be disabled.")

def remove_noise(image):
    """
    Use Non-Local Means Denoising to remove noise.
    Optimized settings for color images.
    """
    try:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    except Exception as e:
        print(f"[ERROR] Noise removal failed: {str(e)}")
        return image  # Return the original image on failure

def deblur_image(image):
    """
    Apply a sharpening filter to enhance image clarity.
    Uses a simple kernel-based sharpening technique.
    """
    try:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)
    except Exception as e:
        print(f"[ERROR] Deblurring failed: {str(e)}")
        return image

def upscale_image(image):
    """
    Enhance resolution using AI-based super-resolution (EDSR).
    If the model is not available, the function returns the original image.
    """
    try:
        if sr.getModelName() != "":
            return sr.upsample(image)
        else:
            print("[WARNING] Super-Resolution model not loaded. Returning original image.")
            return image
    except Exception as e:
        print(f"[ERROR] Upscaling failed: {str(e)}")
        return image
