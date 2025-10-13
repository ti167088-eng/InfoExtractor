
# extractor_easyocr.py
import re
import os
import gc
import cv2
import json
import time
import fitz
import logging
import easyocr
import numpy as np

from os import path
from PIL import Image
from langchain.schema import Document
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Utilities / preprocessing
# -------------------------
def deskew_image(cv_img):
    """
    Robust deskew using Canny edges to find text/line pixels.
    Expects a grayscale uint8 image (h,w).
    Returns rotated image (same dtype).
    """
    if cv_img is None or cv_img.size == 0:
        return cv_img
    # if colored, convert
    if len(cv_img.shape) == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_img.copy()

    # use Canny edges (less sensitive to background)
    edges = cv2.Canny(gray, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    if len(coords) < 50:
        # not enough edges to compute rotation robustly
        return cv_img

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # ignore tiny angles
    if abs(angle) < 0.5:
        return cv_img

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def clean_extracted_text(text: str) -> str:
    """Minimal safe cleaning for OCR text before passing to NLP/LLM."""
    if not text:
        return text
    # fix common hyphenation across lines: "exam-\nple" -> "example"
    text = re.sub(r'([A-Za-z])-\n([A-Za-z])', r'\1\2', text)
    # normalize CR/LF and collapse excessive newlines
    text = text.replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    # trim spaces per line
    lines = [ln.strip() for ln in text.splitlines()]
    # drop empty lines at boundaries
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop(-1)
    return "\n".join(lines)

def clean_image_for_ocr(pil_img, upscale_min_height=900):
    """
    Preprocess image for OCR:
    - convert to grayscale
    - CLAHE (contrast)
    - denoise
    - deskew using edges
    - upscale small images (so OCR has enough pixels)
    Returns a PIL.Image in RGB mode (many OCR engines accept color arrays)
    """
    cv_img = np.array(pil_img)
    # quick fallback: if blank/very white, return original
    if cv_img.mean() > 250:
        return pil_img

    # grayscale
    if len(cv_img.shape) == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv_img.copy()

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # denoise (fastNlMeans)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10)

    # deskew via edges
    deskewed = deskew_image(gray)

    # upscale if very small
    h, w = deskewed.shape[:2]
    if h < upscale_min_height:
        scale = min(2.0, upscale_min_height / float(max(1, h)))
        deskewed = cv2.resize(deskewed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # return as RGB (some engines perform better with color input)
    rgb = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)

# -------------------------
# OCR helpers (word-level with EasyOCR)
# -------------------------
def image_to_words_easyocr(pil_img, reader):
    """
    Runs EasyOCR and returns:
    - page_text (constructed from words, with spaces)
    - words: list of dicts {text, conf (0..100), bbox=(x,y,w,h), start,end}
    - mean_conf (0..100)
    This function builds character offsets by joining words with spaces consistently.
    """
    try:
        # Convert PIL to numpy array for EasyOCR
        img_array = np.array(pil_img)
        
        # EasyOCR readtext returns list of [bbox, text, confidence]
        results = reader.readtext(img_array)
    except Exception as e:
        logger.warning(f"EasyOCR readtext failed: {e}")
        return "", [], 0.0

    words = []
    char_ptr = 0
    confs = []
    
    for detection in results:
        bbox_coords, txt, conf = detection
        txt = txt.strip()
        if not txt:
            continue
            
        # EasyOCR bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # Convert to (left, top, width, height)
        x_coords = [point[0] for point in bbox_coords]
        y_coords = [point[1] for point in bbox_coords]
        left = int(min(x_coords))
        top = int(min(y_coords))
        width = int(max(x_coords) - min(x_coords))
        height = int(max(y_coords) - min(y_coords))
        
        # Convert confidence to 0-100 scale
        confidence = float(conf * 100)
        
        start = char_ptr
        token_text = txt
        char_ptr += len(token_text) + 1
        end = char_ptr - 1
        
        words.append({
            "text": token_text, 
            "conf": confidence, 
            "bbox": (left, top, width, height), 
            "start": start, 
            "end": end
        })
        confs.append(confidence)

    page_text = " ".join([w['text'] for w in words])
    mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
    return page_text, words, mean_conf

# -------------------------
# Main extractor class
# -------------------------
class PDFExtractorEasyOCR:
    def __init__(self, ocr_confidence_threshold=70, num_workers=4, languages=['en'], gpu=False, process_only_do_pdfs=True):
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.num_workers = num_workers
        self.languages = languages
        self.gpu = gpu
        self.process_only_do_pdfs = process_only_do_pdfs
        # Initialize EasyOCR reader
        logger.info(f"Initializing EasyOCR with languages: {languages}, GPU: {gpu}")
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def is_do_pdf(self, pdf_path: str) -> bool:
        """Check if PDF filename contains 'DO' anywhere - very inclusive"""
        filename = path.basename(pdf_path)
        return 'do' in filename.lower()

    def needs_ocr(self, page):
        """
        Determine if a page needs OCR based on text content quality.
        """
        try:
            text = page.get_text().strip()
            if not text or len(text) < 50:
                return True
            # Check if text seems garbled (high ratio of special chars)
            special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
            if special_chars / len(text) > 0.3:
                return True
            return False
        except Exception:
            return True

    def process_page(self, page_index, pdf_path, debug_dir=None):
        """
        Process a single page index from PDF at pdf_path:
        returns langchain.schema.Document with:
          - page_content: cleaned page text (string)
          - metadata: includes page, source_pdf, ocr flag, ocr_mean_conf, words (list)
        """
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_index]
            try:
                text_layer = page.get_text().strip()
            except Exception:
                text_layer = ""

            metadata = {
                "page": page_index + 1,
                "source_pdf": pdf_path,
                "ocr": False,
                "header_footer_removed": False
            }

            # decide whether to OCR: prefer page text if it's long & rich
            if text_layer and not self.needs_ocr(page):
                metadata.update({"ocr": False})
                return Document(page_content=clean_extracted_text(text_layer), metadata=metadata)

            # else rasterize and OCR the page
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            processed_image = clean_image_for_ocr(img)
            # optional debug dump
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, f"page_{page_index+1:03d}_preocr_easyocr.png")
                try:
                    processed_image.save(debug_path)
                except Exception:
                    pass

            page_text, words, mean_conf = image_to_words_easyocr(processed_image, self.reader)
            page_text = clean_extracted_text(page_text)
            metadata.update({
                "ocr": True,
                "confidence": mean_conf,
                "ocr_engine": "EasyOCR",
                "words": words
            })
            return Document(page_content=page_text, metadata=metadata)

        finally:
            doc.close()
            gc.collect()

    def text_extractor(self, pdf_path: str, debug_dir: str = None):
        """
        Orchestrates page processing in parallel using ProcessPoolExecutor.
        Returns a list of Document objects (one per page, in order).
        Only processes DO PDFs if process_only_do_pdfs=True.
        """
        # Check if we should skip non-DO PDFs
        if self.process_only_do_pdfs and not self.is_do_pdf(pdf_path):
            logger.info(f"Skipping non-DO PDF: {pdf_path}")
            return [Document(
                page_content="",
                metadata={
                    "page": 1,
                    "source_pdf": pdf_path,
                    "skipped": True,
                    "reason": "Not a DO PDF"
                }
            )]

        base_doc = fitz.open(pdf_path)
        total_pages = len(base_doc)
        logger.info(f"Processing DO PDF with EasyOCR - Total pages: {total_pages}")
        base_doc.close()

        documents = [None] * total_pages
        start_time = time.time()

        # Note: EasyOCR reader is not picklable, so we process sequentially
        # For parallel processing, each worker would need to initialize its own reader
        for i in range(total_pages):
            try:
                doc = self.process_page(i, pdf_path, debug_dir)
                documents[i] = doc
                logger.info(f"Processed page {i + 1}/{total_pages} (ocr={doc.metadata.get('ocr')}, conf={doc.metadata.get('confidence', None)})")
            except Exception as e:
                logger.error(f"Error on page {i + 1}: {e}")
                documents[i] = Document(
                    page_content="",
                    metadata={"page": i + 1, "error": str(e)}
                )

        elapsed = time.time() - start_time
        logger.info(f"Extraction completed in {elapsed:.1f}s ({total_pages / (elapsed/60):.1f} pages/min)")
        return documents

# -------------------------
# CLI/testing harness
# -------------------------
if __name__ == "__main__":
    sample_pdf = r"C:\Users\pc\Downloads\old pdf\Mojo Leads-20250917T091038Z-1-001\Mojo Leads\Moiz PART B ORDERS\Ada_Manwarren\Ada Manwarren Elbow Cn RX.pdf"
    extractor = PDFExtractorEasyOCR(languages=['en'], gpu=False)
    # set debug_dir to inspect processed page images
    docs = extractor.text_extractor(sample_pdf, debug_dir="debug_ocr_easyocr")
    print(f"Extracted {len(docs)} documents")
    for i, d in enumerate(docs[:3]):
        print(f"\nPage {i+1} Content (first 200 chars): {d.page_content[:200]!r}")
        print(f"Metadata: {json.dumps(d.metadata, indent=2) if d.metadata else None}")

    os.makedirs("results", exist_ok=True)
    with open("results/extractor_easyocr.json", "w", encoding="utf-8") as f:
        json.dump({
            "Extracted Documents": len(docs),
            "Documents": [
                {"Page": i+1, "Content": docs[i].page_content, "Metadata": docs[i].metadata}
                for i in range(len(docs))
            ]
        }, f, indent=2, ensure_ascii=False)
