"""Các hàm tiền xử lý ảnh dermoscopy dùng trong notebook nghiên cứu."""

from __future__ import annotations

import cv2
import numpy as np


def remove_hair(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Phát hiện tóc bằng black-hat morphology và loại tóc bằng inpainting."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    cleaned = cv2.inpaint(image_rgb, mask, 3, cv2.INPAINT_TELEA)
    return cleaned, mask


def apply_clahe(image_rgb: np.ndarray) -> np.ndarray:
    """Tăng tương phản cục bộ trên kênh sáng của không gian LAB."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(lightness)
    merged = cv2.merge((enhanced, channel_a, channel_b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def preprocess_image(
    image_rgb: np.ndarray,
    image_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Khử nhiễu, loại tóc, CLAHE, resize và chuẩn hóa về [0, 1]."""
    denoised = cv2.fastNlMeansDenoisingColored(image_rgb, None, 5, 5, 7, 21)
    hair_removed, _ = remove_hair(denoised)
    enhanced = apply_clahe(hair_removed)
    resized = cv2.resize(enhanced, image_size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

