import cv2
import numpy as np
import albumentations as A
from PIL import Image
import io


def augment_image(image_bytes: bytes) -> bytes:
    """
    Augment the image using various techniques:
    1. Random rotation
    2. Random brightness and contrast
    3. Random horizontal flip
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Define augmentation pipeline
    transform = A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.GaussNoise(p=1),
                    A.GaussianBlur(p=1),
                ],
                p=0.3,
            ),
        ]
    )

    # Apply augmentation
    augmented = transform(image=img)["image"]

    # Convert back to bytes
    success, encoded_img = cv2.imencode(".png", augmented)
    return encoded_img.tobytes()
