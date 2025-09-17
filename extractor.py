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
from PIL import Image
from collections import Counter
from langchain.schema import Document
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def ocr_page_with_retry(image, retries=3):
    for attempt in range(retries):
        try:
            processed_image = clean_image_for_ocr(image)
            data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT, config="--psm 6")
            return data
        except Exception as e:
            logger.warning(f"OCR attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    logger.error("OCR failed after retries")
    return None

class PDFExtractor:
    def __init__(self, ocr_confidence_threshold=70, num_workers=4):
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
                "header_footer_removed": False
            }

            if text and not self.needs_ocr(page):
                return Document(page_content=text, metadata=metadata)

            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            processed_image = clean_image_for_ocr(img)
            ocr_text = pytesseract.image_to_string(processed_image)
            metadata.update({"ocr": True})
            return Document(page_content=ocr_text, metadata=metadata)

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
                    doc = future.result(timeout=120)
                    documents[page_index] = doc
                    logger.info(f"Processed page {page_index + 1}/{total_pages}")
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
        return documents

if __name__ == "__main__":
    path = r"C:\Users\pc\Downloads\Ali Haider -20250917T091038Z-1-002\Ali Haider\Medical Records\Mikhail_Rudin\Mikhail Rudin CN (1).pdf"
    extractor = PDFExtractor()
    documents = extractor.text_extractor(path)
    print(f"Extracted {len(documents)} documents")
    for i, doc in enumerate(documents[:3]):
        print(f"\nPage {i + 1} Content: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")

    os.makedirs("results", exist_ok=True)
    with open("results/extractor_summary.json", "w") as f:
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
