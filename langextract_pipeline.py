# langchain_extract_pipeline.py
"""
Use LangChain structured output + Google Gemini (via langchain_google_genai) to extract
patient_name, date_of_birth and address from the documents produced by extractor.PDFExtractor.

Requirements:
  pip install langchain langchain-google-genai
  (and any deps your extractor requires)

Environment:
  - Set Google credentials / API key required by langchain-google-genai / Gemini:
      export GOOGLE_API_KEY="YOUR_KEY"
  - Or follow the langchain-google-genai docs for auth.
"""

import os
import json
from typing import List, Dict, Any

# Import your extractor (expects extractor.py present)
from extractor import PDFExtractor

# LangChain structured output imports
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Google Gemini / LangChain Google integration
# Install: pip install langchain-google-genai
from langchain_google_genai import ChatGoogleGenerativeAI

# ----------------------
# Config
# ----------------------
PDF_PATH = "path/to/your.pdf"  # change as needed
DEBUG_DIR = "debug_ocr"
OUTPUT_DIR = "langchain_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Choose model (Gemini family)
GEMINI_MODEL = os.environ.get("LANGCHAIN_GEMINI_MODEL", "gemini-2.5-flash")

# ----------------------
# Helper: docs -> text (per-page recommended)
# ----------------------
def docs_to_page_texts(documents: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert langchain Document list (from your PDFExtractor) into per-page dicts:
      {"id": "page_1", "text": "...", "meta": {"page": 1, "source": PDF_PATH}}
    """
    out = []
    for i, d in enumerate(documents):
        page_no = d.metadata.get("page", i + 1)
        out.append({
            "id": f"page_{page_no}",
            "text": d.page_content or "",
            "meta": {"page": page_no, "source": getattr(d, "metadata", {}).get("source", PDF_PATH)}
        })
    return out

# ----------------------
# Build StructuredOutputParser + Prompt
# ----------------------
def build_parser_and_prompt():
    # Define response schema fields (name + short description)
    response_schemas = [
        ResponseSchema(name="patient_name", description="Full patient name (string)"),
        ResponseSchema(name="date_of_birth", description="Date of birth, MM/DD/YYYY if available (string)"),
        ResponseSchema(name="address", description="Street, city, state, zip if available (string)")
    ]

    parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # The parser gives format instructions to include in the prompt
    format_instructions = parser.get_format_instructions()

    # Few-shot examples (short) — using your examples
    few_shot_examples = [
        """Example 1:
Name: HESS, MYRTLE K
DOB: 08/25/1939
1 Clinic Drive, RICHLANDS, VA 24641-1102
Phone: (276) 964-2281
id #686725
-> patient_name: HESS, MYRTLE K
   date_of_birth: 08/25/1939
   address: 1 Clinic Drive, RICHLANDS, VA 24641-1102
""",
        """Example 2:
-DOB-08/25/1939 | Address: 112 MAY ST City: RICHLANDS | Patient Phone Number: 2769646189 - Phone Number: 2769641281 | Fax Number: 2769641373
-> patient_name: null
   date_of_birth: 08/25/1939
   address: 112 MAY ST, RICHLANDS
""",
        """Example 3:
CHRISTINE L ALLEN
5314 20th St N
Kalamazoo MI 49004
Date: 08/04/2025
-> patient_name: CHRISTINE L ALLEN
   date_of_birth: null
   address: 5314 20th St N, Kalamazoo, MI 49004
"""
    ]

    # Prompt template: we give the format instructions + examples + the page text to extract from.
    prompt_template = """You are an extraction assistant. Extract the following fields from the INPUT_TEXT:
{format_instructions}

Few-shot examples (examples show the expected extraction format):
{few_shot}

Now extract fields (patient_name, date_of_birth, address) from the INPUT_TEXT below.
If a field is not present, return null for that field.

INPUT_TEXT:
{input_text}

Answer in the format specified above (JSON code block).
"""

    prompt = PromptTemplate(
        input_variables=["format_instructions", "few_shot", "input_text"],
        template=prompt_template
    )

    return parser, prompt, "\n\n".join(few_shot_examples)

# ----------------------
# Run extraction on documents
# ----------------------
def run_extraction(documents: List[Any]):
    # Convert to per-page texts
    pages = docs_to_page_texts(documents)

    # Build parser + prompt + few-shot examples
    parser, prompt_template, few_shot_text = build_parser_and_prompt()
    format_instructions = parser.get_format_instructions()

    # Initialize Gemini chat model (LangChain wrapper)
    # Make sure your environment auth is set per langchain-google-genai docs
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL)

    all_results = []

    # Process each page (per-page for precise grounding)
    for p in pages:
        input_text = p["text"].strip() or ""
        if not input_text:
            continue

        prompt_str = prompt_template.format(
            format_instructions=format_instructions,
            few_shot=few_shot_text,
            input_text=input_text
        )

        # ChatGoogleGenerativeAI expects messages; wrap as a single human message
        hm = HumanMessage(content=prompt_str)

        # Call model
        resp = llm.invoke([hm])
        raw = resp.content

        # Parse model output using the StructuredOutputParser
        try:
            parsed = parser.parse(raw)
        except Exception as e:
            # parsing failure; store raw and error
            parsed = {"_parse_error": str(e), "_raw": raw}

        out = {
            "page_id": p["id"],
            "page_no": p["meta"].get("page"),
            "raw": raw,
            "parsed": parsed
        }
        all_results.append(out)

        # OPTIONAL: save per-page JSON for inspection
        with open(os.path.join(OUTPUT_DIR, f"{p['id']}_extraction.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    # Save combined results
    with open(os.path.join(OUTPUT_DIR, "combined_extractions.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results

# ----------------------
# CLI main
# ----------------------
def main():
    if not os.path.exists(PDF_PATH):
        print(f"PDF not found at {PDF_PATH}. Update PDF_PATH and retry.")
        return

    print(f"Processing PDF: {PDF_PATH}")
    extractor = PDFExtractor(ocr_confidence_threshold=70, num_workers=4)
    docs = extractor.text_extractor(PDF_PATH, debug_dir=DEBUG_DIR)
    print(f"Extracted {len(docs)} page documents")

    print("Running LangChain + Gemini extraction...")
    results = run_extraction(docs)
    print(f"Saved extraction files to: {OUTPUT_DIR}")
    return results

if __name__ == "__main__":
    main()
