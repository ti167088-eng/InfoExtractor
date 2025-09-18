# langextract_pipeline.py
"""
Pipeline: use your extractor.py to get page texts (langchain.Document objects),
then run LangExtract to pull structured patient fields: name, dob, address, phone, mrn, etc.

Place this file next to extractor.py. It expects the extractor to return a list
of langchain.schema.Document objects (the same as your extractor.text_extractor()).
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

# import your extractor (assumes extractor.py defines PDFExtractor)
from extractor import PDFExtractor

# LangExtract imports
# install via: pip install langextract
try:
    from langextract import extract, ExampleData
except ImportError:
    print("LangExtract not installed. Run: pip install langextract")
    exit(1)

# ---------------------------
# 1) Configuration
# ---------------------------
PDF_PATH = "C:\\Users\\pc\\Downloads\\Mojo Leads-20250917T091038Z-1-001\\Mojo Leads\\Moiz PPO Orders\\Norman Gibney\\Norman Gibney Rx +LMN.pdf"  # change this to your actual PDF path
DEBUG_DIR = "debug_ocr"
OUTPUT_DIR = "langextract_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------
# 2) Small helper to convert langchain Document -> single string with page markers
# ---------------------------
def docs_to_text_blocks(documents: List[Any],
                        mode: str = "concat") -> List[Dict[str, Any]]:
    """
    Convert your extractor's list of Document objects into blocks LangExtract can consume.
    mode:
      - "concat": combine all pages into one big string (with page separators).
      - "per_page": return one block per page (useful if you want per-page grounding).
    Returns list of {"id": str, "text": str, "meta": {...}}
    """
    blocks = []
    if mode == "concat":
        all_text = []
        for d in documents:
            page_no = d.metadata.get("page", None)
            header = f"\n\n--- PAGE {page_no} ---\n\n" if page_no else "\n\n--- PAGE ---\n\n"
            all_text.append(header + (d.page_content or ""))
        blocks.append({
            "id": "document_1",
            "text": "\n".join(all_text),
            "meta": {
                "source": PDF_PATH
            }
        })
    else:
        for i, d in enumerate(documents):
            blocks.append({
                "id": f"page_{i+1}",
                "text": d.page_content or "",
                "meta": {
                    "page": d.metadata.get("page"),
                    "source": PDF_PATH
                }
            })
    return blocks


# ---------------------------
# 3) Define LangExtract schema + few-shot example(s)
# ---------------------------
def build_schema_and_examples():
    """
    Define expected fields for patient data extraction using langextract.
    """
    # Define schema using langextract format
    schema = {
        "patient_name": "Full patient name",
        "date_of_birth": "Patient date of birth", 
        "address": "Patient address (street, city, state, zip if available)",
        "phone": "Contact phone number",
        "medical_record_number": "MRN or patient ID if present",
        "gender": "Patient gender (Male/Female/Other/Unknown)",
    }

    # Create examples as required by langextract
    examples = [
        ExampleData(
            text="""Name: HESS, MYRTLE K
DOB: 08/25/1939
1 Clinic Drive, RICHLANDS, VA 24641-1102
Phone: (276) 964-2281
id #686725""",
            output={
                "patient_name": "HESS, MYRTLE K",
                "date_of_birth": "08/25/1939",
                "address": "1 Clinic Drive, RICHLANDS, VA 24641-1102",
                "phone": "(276) 964-2281",
                "medical_record_number": "686725",
                "gender": None
            }
        ),
        ExampleData(
            text="""-DOB-08/25/1939 | Address: 112 MAY ST City: RICHLANDS | Patient Phone Number: 2769646189 - Phone Number: 2769641281 | Fax Number: 2769641373""",
            output={
                "patient_name": None,
                "date_of_birth": "08/25/1939",
                "address": "112 MAY ST, RICHLANDS, VA 24641",
                "phone": "2769646189",
                "medical_record_number": None,
                "gender": None
            }
        ),
        ExampleData(
            text="""CHRISTINE L ALLEN
5314 20th St N
Kalamazoo MI 49004
Date: 08/04/2025""",
            output={
                "patient_name": "CHRISTINE L ALLEN",
                "date_of_birth": None,
                "address": "5314 20th St N, Kalamazoo, MI 49004",
                "phone": None,
                "medical_record_number": None,
                "gender": None
            }
        )
    ]

    return schema, examples


# ---------------------------
# 4) Run pipeline
# ---------------------------
def run_langextract_on_documents(documents):
    """
    Main extraction pipeline using LangExtract
    """
    # Prepare text blocks
    blocks = docs_to_text_blocks(documents, mode="concat")

    # Build schema and examples
    schema, examples = build_schema_and_examples()

    # Get the combined text from all blocks
    combined_text = "\n\n".join([block["text"] for block in blocks])

    # Run langextract extraction with proper API
    try:
        result = extract(
            data=combined_text,
            schema=schema,
            examples=examples
        )
    except Exception as e:
        print(f"LangExtract extraction failed: {e}")
        result = {"error": str(e)}

    # Save results
    out_json = os.path.join(OUTPUT_DIR, "patient_extractions.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Extraction JSON written to: {out_json}")
    return result


# ---------------------------
# 5) CLI / main
# ---------------------------
def main():
    """
    Main function to run the complete pipeline
    """
    if not os.path.exists(PDF_PATH):
        print(
            f"PDF not found at {PDF_PATH}. Please update PDF_PATH or provide a valid file."
        )
        return

    # Option A: Run extractor live
    print(f"Processing PDF: {PDF_PATH}")
    extractor = PDFExtractor(ocr_confidence_threshold=70, num_workers=4)
    docs = extractor.text_extractor(PDF_PATH, debug_dir=DEBUG_DIR)
    print(f"Extracted {len(docs)} page documents from {PDF_PATH}")

    # Option B (commented): read from saved extractor JSON (if you previously stored results)
    # with open("results/extractor.json", "r", encoding="utf-8") as f:
    #     saved = json.load(f)
    #     # convert saved JSON back into blocks for LangExtract if desired
    #     # (implement conversion depending on saved schema)

    # Run LangExtract pipeline
    print("Running LangExtract extraction...")
    result = run_langextract_on_documents(docs)

    print("Pipeline completed successfully!")
    return result


if __name__ == "__main__":
    main()