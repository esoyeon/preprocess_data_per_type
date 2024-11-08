import cv2
import numpy as np
from PIL import Image
import io


def preprocess_image(image_bytes: bytes) -> bytes:
    """
    Preprocess the image:
    1. Convert to grayscale
    2. Apply histogram equalization
    3. Apply Gaussian blur for noise reduction
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

    # Convert back to bytes
    success, encoded_img = cv2.imencode(".png", blurred)
    return encoded_img.tobytes()
