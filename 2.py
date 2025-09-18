
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

def enhance_for_numbers(cv_img):
    """Additional preprocessing specifically for number recognition"""
    # Create a copy for number enhancement
    number_enhanced = cv_img.copy()
    
    # Sharpen the image for better digit recognition
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(number_enhanced, -1, kernel)
    
    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(sharpened, 9, 75, 75)
    
    return bilateral

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
    
    # Step 4: Special enhancement for numbers
    number_enhanced = enhance_for_numbers(deskewed)
    
    # Step 5: Adaptive thresholding with multiple methods
    # Try Gaussian adaptive threshold
    thresh1 = cv2.adaptiveThreshold(number_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Try Mean adaptive threshold
    thresh2 = cv2.adaptiveThreshold(number_enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Try Otsu's thresholding for better number recognition
    _, thresh3 = cv2.threshold(number_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine thresholds with weights favoring number clarity
    combined_thresh = cv2.addWeighted(thresh1, 0.4, thresh2, 0.3, 0)
    combined_thresh = cv2.addWeighted(combined_thresh, 0.7, thresh3, 0.3, 0)
    
    # Step 6: Morphological operations optimized for text and numbers
    # Small kernel for cleaning without destroying thin strokes
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel_small)
    
    # Slightly larger kernel for closing gaps in characters
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
    
    # Step 7: Scale up for better OCR (optimized scaling)
    height, width = final.shape
    if height < 2000 or width < 2000:  # Increased target size for better number recognition
        scale_factor = max(2000 / height, 2000 / width, 2.0)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        final = cv2.resize(final, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return Image.fromarray(final)

def extract_text_with_multiple_configs(image):
    """Try multiple OCR configurations optimized for both numbers and text"""
    # Enhanced configs with better number recognition
    configs = [
        # Specialized for numbers and medical data
        "--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()[]{}/'\"-%+= --oem 3",
        "--psm 4 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()[]{}/'\"-%+= --oem 3",
        "--psm 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()[]{}/'\"-%+= --oem 3",
        # Number-focused configs
        "--psm 8 -c tessedit_char_whitelist=0123456789.,/%+- --oem 3",
        "--psm 7 -c tessedit_char_whitelist=0123456789.,/%+- --oem 3",
        # General configs with different OEM modes
        "--psm 6 --oem 1",
        "--psm 4 --oem 1", 
        "--psm 3 --oem 1",
        "--psm 6 --oem 3",
        "--psm 4 --oem 3",
        "--psm 3 --oem 3"
    ]
    
    results = []
    
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
                
                if text:
                    # Count numbers and words for scoring
                    numbers = len(re.findall(r'\d+', text))
                    words = len(re.findall(r'[a-zA-Z]+', text))
                    
                    # Enhanced scoring that values both numbers and words
                    score = (words * 1.0 + numbers * 1.5) * avg_confidence
                    
                    results.append({
                        'text': text,
                        'confidence': avg_confidence,
                        'score': score,
                        'numbers': numbers,
                        'words': words,
                        'config': config
                    })
                    
        except Exception as e:
            logger.warning(f"OCR config failed: {config}, error: {e}")
            continue
    
    if not results:
        return "", 0
    
    # Sort by score and get the best result
    results.sort(key=lambda x: x['score'], reverse=True)
    best_result = results[0]
    
    # If multiple results have similar scores, combine them intelligently
    if len(results) > 1:
        top_results = [r for r in results[:3] if r['score'] > best_result['score'] * 0.8]
        
        # Combine number-rich and word-rich results
        number_rich = max(top_results, key=lambda x: x['numbers'])
        word_rich = max(top_results, key=lambda x: x['words'])
        
        if number_rich != word_rich and number_rich['numbers'] > word_rich['numbers'] * 0.5:
            # Try to intelligently merge
            combined_text = merge_ocr_results(word_rich['text'], number_rich['text'])
            if len(combined_text) > len(best_result['text']) * 0.9:
                return combined_text, (word_rich['confidence'] + number_rich['confidence']) / 2
    
    return best_result['text'], best_result['confidence']

def merge_ocr_results(text1, text2):
    """Intelligently merge two OCR results to get best of both"""
    if not text1 or not text2:
        return text1 or text2
    
    lines1 = text1.split('\n')
    lines2 = text2.split('\n')
    
    merged_lines = []
    max_lines = max(len(lines1), len(lines2))
    
    for i in range(max_lines):
        line1 = lines1[i] if i < len(lines1) else ""
        line2 = lines2[i] if i < len(lines2) else ""
        
        # Choose line with more numbers if one has significantly more
        nums1 = len(re.findall(r'\d+', line1))
        nums2 = len(re.findall(r'\d+', line2))
        
        if nums2 > nums1 * 2:
            merged_lines.append(line2)
        elif nums1 > nums2 * 2:
            merged_lines.append(line1)
        else:
            # Choose longer line
            merged_lines.append(line1 if len(line1) > len(line2) else line2)
    
    return '\n'.join(merged_lines)

def post_process_medical_text(text):
    """Enhanced post-processing for better number and text recognition"""
    if not text:
        return text
    
    # Phase 1: Fix number-specific issues first
    # Protect numbers during text corrections
    number_patterns = re.findall(r'\b\d+\.?\d*\b', text)
    protected_numbers = {}
    
    for i, num in enumerate(number_patterns):
        placeholder = f"__NUM_{i}__"
        protected_numbers[placeholder] = num
        text = text.replace(num, placeholder, 1)
    
    # Phase 2: Context-aware corrections
    # Fix O/0 confusion based on context
    text = re.sub(r'\bO(?=\d)', '0', text)  # O before digits -> 0
    text = re.sub(r'(?<=\d)O\b', '0', text)  # O after digits -> 0
    text = re.sub(r'\b0(?=[A-Z][a-z])', 'O', text)  # 0 before words -> O
    
    # Fix I/1/l confusion based on context
    text = re.sub(r'\b1(?=[a-zA-Z])', 'I', text)  # 1 before letters -> I
    text = re.sub(r'(?<=[a-zA-Z])1\b', 'I', text)  # 1 after letters -> I
    text = re.sub(r'\bl(?=[A-Z])', 'I', text)  # l before uppercase -> I
    text = re.sub(r'\bl\b(?=\s[A-Z])', 'I', text)  # standalone l before uppercase -> I
    
    # Fix S/5 confusion
    text = re.sub(r'\b5(?=[a-z]{2,})', 'S', text)  # 5 before words -> S
    text = re.sub(r'(?<=[a-z])5\b', 's', text)  # 5 after lowercase -> s
    
    # Common medical OCR corrections
    corrections = {
        r'rn\b': 'm',  # rn to m
        r'\brn': 'm',   # rn to m at start
        r'vv': 'w',     # vv to w
        r'VV': 'W',     # VV to W
        r'iii': 'iii',  # Keep medical notation
        r'(?<=[a-zA-Z])\s+(?=[a-zA-Z])(?=\w{1,2}\s)': '',  # Join very short words
        r'[|]': 'I',    # Pipe to I
        r'(?<=\w)[°](?=\w)': 'o',  # Degree symbol to o in middle of words
    }
    
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)
    
    # Phase 3: Restore protected numbers
    for placeholder, original_num in protected_numbers.items():
        text = text.replace(placeholder, original_num)
    
    # Phase 4: Final cleanup for numbers
    # Remove spaces within numbers
    text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)
    text = re.sub(r'(?<=\d)\s*\.\s*(?=\d)', '.', text)  # Fix decimal points
    text = re.sub(r'(?<=\d)\s*,\s*(?=\d{3})', ',', text)  # Fix thousand separators
    
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
    with open("results/2.json", "w") as f:
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
