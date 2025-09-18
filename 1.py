
import re
import os
import gc
import cv2
import json
import time
import fitz
import logging
import pytesseract
import numpy as np

from os import path
from PIL import Image, ImageEnhance, ImageFilter
from collections import Counter
from langchain.schema import Document
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enhance_contrast_and_brightness(pil_img):
    """Enhanced contrast and brightness adjustment for medical documents"""
    # Convert to grayscale if not already
    if pil_img.mode != 'L':
        pil_img = pil_img.convert('L')
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    cv_img = np.array(pil_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(cv_img)
    
    return Image.fromarray(enhanced)

def remove_noise_advanced(cv_img):
    """Advanced noise removal specifically for medical documents"""
    # Apply multiple denoising techniques
    
    # 1. Non-local means denoising
    denoised = cv2.fastNlMeansDenoising(cv_img, None, 10, 7, 21)
    
    # 2. Morphological operations to remove small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
    
    # 3. Remove isolated pixels (salt and pepper noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def detect_and_correct_skew_advanced(cv_img):
    """Advanced skew detection and correction"""
    # Create a copy for processing
    img_copy = cv_img.copy()
    
    # Apply edge detection
    edges = cv2.Canny(img_copy, 50, 150, apertureSize=3)
    
    # Use HoughLines to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is not None:
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            # Focus on near-horizontal lines
            if 85 <= angle <= 95:
                angles.append(angle - 90)
            elif -5 <= angle <= 5:
                angles.append(angle)
        
        if angles:
            # Use median angle to avoid outliers
            median_angle = np.median(angles)
            
            # Only correct if skew is significant
            if abs(median_angle) > 0.5:
                (h, w) = cv_img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(cv_img, M, (w, h), 
                                       flags=cv2.INTER_CUBIC, 
                                       borderMode=cv2.BORDER_REPLICATE)
                return rotated
    
    return cv_img

def preprocess_medical_image(pil_img):
    """Comprehensive preprocessing pipeline for medical documents"""
    # Step 1: Enhance contrast and brightness
    enhanced = enhance_contrast_and_brightness(pil_img)
    
    # Convert to OpenCV format
    cv_img = np.array(enhanced)
    
    # Step 2: Advanced noise removal
    denoised = remove_noise_advanced(cv_img)
    
    # Step 3: Advanced skew correction
    deskewed = detect_and_correct_skew_advanced(denoised)
    
    # Step 4: Adaptive thresholding with multiple methods
    # Try Gaussian adaptive threshold
    thresh1 = cv2.adaptiveThreshold(deskewed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 4)
    
    # Try Mean adaptive threshold
    thresh2 = cv2.adaptiveThreshold(deskewed, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, 4)
    
    # Combine both thresholds
    combined_thresh = cv2.bitwise_and(thresh1, thresh2)
    
    # Step 5: Final morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    final = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)
    
    # Step 6: Scale up for better OCR
    height, width = final.shape
    if height < 1500 or width < 1500:
        scale_factor = max(1500 / height, 1500 / width, 1.5)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        final = cv2.resize(final, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return Image.fromarray(final)

def extract_text_with_multiple_configs(image):
    """Try multiple OCR configurations and return the best result"""
    configs = [
        "--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()[]{}/'\"- ",
        "--psm 4 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()[]{}/'\"- ",
        "--psm 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()[]{}/'\"- ",
        "--psm 6",
        "--psm 4",
        "--psm 3"
    ]
    
    best_text = ""
    best_confidence = 0
    
    for config in configs:
        try:
            # Get text with confidence data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                
                # Get the text
                text = pytesseract.image_to_string(image, config=config).strip()
                
                # Prefer longer text with decent confidence
                score = len(text.split()) * avg_confidence
                
                if score > best_confidence * len(best_text.split()):
                    best_text = text
                    best_confidence = avg_confidence
                    
        except Exception as e:
            logger.warning(f"OCR config failed: {config}, error: {e}")
            continue
    
    return best_text, best_confidence

def post_process_medical_text(text):
    """Post-process OCR text to fix common medical document issues"""
    if not text:
        return text
    
    # Common OCR corrections for medical documents
    corrections = {
        r'\b0\b': 'O',  # Zero to O
        r'\bO\b(?=\d)': '0',  # O to zero when followed by digits
        r'\bl\b': 'I',  # lowercase l to uppercase I
        r'\b1\b(?=[a-zA-Z])': 'I',  # 1 to I when followed by letters
        r'rn\b': 'm',  # rn to m
        r'\brn': 'm',   # rn to m at start
        r'vv': 'w',     # vv to w
        r'VV': 'W',     # VV to W
        r'(?<=\d)\s+(?=\d)': '',  # Remove spaces between digits
        r'(?<=[a-zA-Z])\s+(?=[a-zA-Z])(?=\w{1,3}\s)': '',  # Join short words
    }
    
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text.strip()

class PDFExtractor:
    def __init__(self, ocr_confidence_threshold=60, num_workers=4):
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.num_workers = num_workers

    def needs_ocr(self, page):
        text = page.get_text().strip()
        if len(text) > 100 and re.search(r'\w+\s+\w+', text):
            return False
        if len(page.get_images()) > 3 and len(text) < 20:
            return False
        return True

    def process_page(self, page_index, path):
        doc = fitz.open(path)
        try:
            page = doc[page_index]
            text = page.get_text().strip()
            metadata = {
                "page": page_index + 1,
                "source_pdf": path,
                "ocr": False,
                "header_footer_removed": False,
                "confidence": None
            }

            if text and not self.needs_ocr(page):
                return Document(page_content=text, metadata=metadata)

            # Enhanced image extraction with higher DPI
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x scaling for better quality
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Apply enhanced preprocessing
            processed_image = preprocess_medical_image(img)
            
            # Extract text with multiple configurations
            ocr_text, confidence = extract_text_with_multiple_configs(processed_image)
            
            # Post-process the text
            cleaned_text = post_process_medical_text(ocr_text)
            
            metadata.update({
                "ocr": True,
                "confidence": confidence
            })
            
            return Document(page_content=cleaned_text, metadata=metadata)

        finally:
            doc.close()
            gc.collect()

    def text_extractor(self, path: str):
        base_doc = fitz.open(path)
        total_pages = len(base_doc)
        logger.info(f"Total pages: {total_pages}")
        base_doc.close()

        documents = [None] * total_pages
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=min(self.num_workers, total_pages)) as executor:
            futures = {executor.submit(self.process_page, i, path): i for i in range(total_pages)}
            for future in as_completed(futures):
                page_index = futures[future]
                try:
                    doc = future.result(timeout=180)  # Increased timeout for complex processing
                    documents[page_index] = doc
                    confidence_info = f" (confidence: {doc.metadata.get('confidence', 'N/A')})" if doc.metadata.get('ocr') else ""
                    logger.info(f"Processed page {page_index + 1}/{total_pages}{confidence_info}")
                except TimeoutError:
                    logger.error(f"Timeout on page {page_index + 1}")
                    documents[page_index] = Document(
                        page_content="",
                        metadata={"page": page_index + 1, "error": "Timeout"}
                    )
                except Exception as e:
                    logger.error(f"Error on page {page_index + 1}: {e}")
                    documents[page_index] = Document(
                        page_content="",
                        metadata={"page": page_index + 1, "error": str(e)}
                    )

        elapsed = time.time() - start_time
        logger.info(f"Extraction completed in {elapsed:.1f}s ({total_pages / (elapsed/60):.1f} pages/min)")
        
        # Log confidence statistics
        confidences = [doc.metadata.get('confidence') for doc in documents if doc and doc.metadata.get('confidence')]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            logger.info(f"Average OCR confidence: {avg_confidence:.1f}%")
        
        return documents

if __name__ == "__main__":
    path = r"C:\Users\pc\Downloads\Mojo Leads-20250917T091038Z-1-001\Mojo Leads\Moiz PART B ORDERS\SAMUEL COLES\SAMUEL COLES CN AND RX BT HIPS (1).pdf"
    extractor = PDFExtractor()
    documents = extractor.text_extractor(path)
    print(f"Extracted {len(documents)} documents")
    for i, doc in enumerate(documents[:3]):
        print(f"\nPage {i + 1} Content: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")

    os.makedirs("results", exist_ok=True)
    with open("results/1.json", "w") as f:
        json.dump({
            "Extracted Documents": len(documents),
            "Documents": [
                {
                    "Page": i + 1,
                    "Content": doc.page_content,
                    "Metadata": doc.metadata
                } for i, doc in enumerate(documents)
            ]
        }, f, indent=4)
