
import re
import os
import gc
import cv2
import json
import time
import fitz
import logging
import numpy as np

from os import path
from PIL import Image
from collections import Counter
from langchain.schema import Document
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global PaddleOCR instance for multiprocessing
_paddle_ocr = None

def get_paddle_ocr():
    """Get or create PaddleOCR instance for current process"""
    global _paddle_ocr
    if _paddle_ocr is None:
        _paddle_ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en'
        )
    return _paddle_ocr

def deskew_image(cv_img):
    coords = np.column_stack(np.where(cv_img > 0))
    if len(coords) < 5:
        return cv_img  # nothing to deskew
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = cv_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def clean_image_for_ocr(pil_img):
    cv_img = np.array(pil_img)
    if cv_img.mean() > 200:
        return pil_img
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    # 🔥 Bilateral filtering to smooth noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # 🔥 Adaptive threshold to handle uneven backgrounds
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # 🔥 Morphological opening to remove small noise
    kernel = np.ones((2,2), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 🔥 Deskew to straighten text lines
    deskewed = deskew_image(opened)

    # 🔥 Scale up if small
    if cv_img.shape[0] < 1000:
        scaled = cv2.resize(deskewed, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    else:
        scaled = deskewed

    return Image.fromarray(scaled)

def extract_text_with_paddle(image):
    """Extract text using PaddleOCR with proper error handling"""
    try:
        # Get PaddleOCR instance for this process
        paddle_ocr = get_paddle_ocr()
        
        # Convert PIL image to numpy array for PaddleOCR
        img_array = np.array(image)
        
        # Use PaddleOCR to extract text
        result = paddle_ocr.ocr(img_array)
        
        # Handle empty or None results
        if not result:
            logger.debug("PaddleOCR returned None result")
            return "", 0
            
        # Handle case where result is a list but empty or contains None
        if isinstance(result, list):
            if len(result) == 0:
                logger.debug("PaddleOCR returned empty list")
                return "", 0
            if result[0] is None:
                logger.debug("PaddleOCR returned list with None")
                return "", 0
            
            # PaddleOCR returns nested lists - get the actual OCR data
            ocr_data = result[0]
        else:
            ocr_data = result
        
        # Extract text and confidence from results
        extracted_text = []
        confidences = []
        
        for line in ocr_data:
            if line and len(line) >= 2:
                try:
                    # Structure: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], [text, confidence]]
                    coords, text_data = line[0], line[1]
                    
                    if isinstance(text_data, list) and len(text_data) >= 2:
                        text = str(text_data[0]).strip()  # Text content
                        confidence = float(text_data[1])  # Confidence score
                        
                        if text:  # Only add non-empty text
                            extracted_text.append(text)
                            confidences.append(confidence * 100)  # Convert to percentage
                    elif isinstance(text_data, str):
                        # Some versions return just text without confidence
                        text = text_data.strip()
                        if text:
                            extracted_text.append(text)
                            confidences.append(100.0)  # Default confidence
                            
                except (IndexError, TypeError, ValueError) as e:
                    logger.debug(f"Error parsing OCR line {line}: {e}")
                    continue
        
        # Combine all text
        full_text = '\n'.join(extracted_text)
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        logger.debug(f"Extracted {len(extracted_text)} lines with avg confidence {avg_confidence:.1f}%")
        
        return full_text, avg_confidence
        
    except Exception as e:
        logger.warning(f"PaddleOCR extraction failed: {e}")
        return "", 0

def post_process_medical_text(text):
    """Clean and post-process extracted text"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might be OCR artifacts
    text = re.sub(r'[^\w\s.,;:!?()[\]{}/\'"%-+=]', '', text)
    
    # Fix common OCR mistakes
    text = text.replace('|', 'I')
    text = text.replace('0', 'O')  # Context-dependent, might need refinement
    
    return text.strip()

def process_single_page(page_data):
    """Process a single page - used by multiprocessing"""
    page_index, pdf_path = page_data
    
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        text = page.get_text().strip()
        metadata = {
            "page": page_index + 1,
            "source_pdf": pdf_path,
            "ocr": False,
            "header_footer_removed": False,
            "confidence": None,
            "ocr_engine": "PaddleOCR"
        }

        # Check if page needs OCR
        needs_ocr = True
        if len(text) > 100 and re.search(r'\w+\s+\w+', text):
            needs_ocr = False
        elif len(page.get_images()) == 0 and len(text) > 50:
            needs_ocr = False

        if text and not needs_ocr:
            return Document(page_content=text, metadata=metadata)

        # Enhanced image extraction with higher DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x scaling for better quality
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Apply enhanced preprocessing
        processed_image = clean_image_for_ocr(img)
        
        # Extract text with PaddleOCR
        ocr_text, confidence = extract_text_with_paddle(processed_image)
        
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

class PaddlePDFExtractor:
    def __init__(self, ocr_confidence_threshold=60, num_workers=4):
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.num_workers = num_workers

    def text_extractor(self, path: str):
        base_doc = fitz.open(path)
        total_pages = len(base_doc)
        logger.info(f"Total pages: {total_pages}")
        base_doc.close()

        documents = [None] * total_pages
        start_time = time.time()

        # Prepare page data for multiprocessing
        page_data = [(i, path) for i in range(total_pages)]

        with ProcessPoolExecutor(max_workers=min(self.num_workers, total_pages)) as executor:
            futures = {executor.submit(process_single_page, data): data[0] for data in page_data}
            
            for future in as_completed(futures):
                page_index = futures[future]
                try:
                    doc = future.result(timeout=180)  # Increased timeout for complex processing
                    documents[page_index] = doc
                    confidence_info = f" (confidence: {doc.metadata.get('confidence', 'N/A'):.1f}%)" if doc.metadata.get('ocr') else ""
                    logger.info(f"Processed page {page_index + 1}/{total_pages}{confidence_info}")
                except TimeoutError:
                    logger.error(f"Timeout on page {page_index + 1}")
                    documents[page_index] = Document(
                        page_content="",
                        metadata={"page": page_index + 1, "error": "Timeout", "ocr_engine": "PaddleOCR"}
                    )
                except Exception as e:
                    logger.error(f"Error on page {page_index + 1}: {e}")
                    documents[page_index] = Document(
                        page_content="",
                        metadata={"page": page_index + 1, "error": str(e), "ocr_engine": "PaddleOCR"}
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
    extractor = PaddlePDFExtractor()
    documents = extractor.text_extractor(path)
    print(f"Extracted {len(documents)} documents")
    for i, doc in enumerate(documents[:3]):
        print(f"\nPage {i + 1} Content: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")

    os.makedirs("results", exist_ok=True)
    with open("results/paddle_extractor_summary.json", "w") as f:
        json.dump({
            "Extracted Documents": len(documents),
            "OCR Engine": "PaddleOCR",
            "Documents": [
                {
                    "Page": i + 1,
                    "Content": doc.page_content,
                    "Metadata": doc.metadata
                } for i, doc in enumerate(documents)
            ]
        }, f, indent=4)
