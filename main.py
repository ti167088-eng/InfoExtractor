
#!/usr/bin/env python3
"""
Main script to process PDF files and extract structured data.
"""

import json
import sys
import os
from data_parser import DataParser

def main():
    """Main function to process PDF and extract data"""
    
    # Initialize the parser (which includes the PDF extractor)
    parser = DataParser()
    
    # Example: Process the sample PDF if provided as command line argument
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Hardcode your PDF path here for testing
        pdf_path = r"C:\Users\pc\Downloads\old pdf\Ali Haider -20250917T091038Z-1-002\Ali Haider\Medical Records\Donald _Hagen\Donald Hagen  DO.pdf"  # Replace with your actual PDF path
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        print("\nPDF Data Extractor")
        print("=" * 50)
        print("Usage: python main.py <path_to_pdf_file>")
        print("\nThis script will:")
        print("1. Extract text from the PDF using OCR if needed")
        print("2. Parse the text to find structured data like:")
        print("   - First Name, Last Name")
        print("   - Date of Birth")
        print("   - Address, City, State, Postal Code")
        print("   - Phone Number")
        print("   - Physician Name, NPI")
        print("   - Diagnosis codes")
        return
    
    print(f"Processing PDF: {pdf_path}")
    print("=" * 50)
    
    # Process the PDF
    result = parser.process_pdf_file(pdf_path, debug_dir="debug_output")
    
    if result.get("error"):
        print(f"Error processing PDF: {result['error']}")
    else:
        print(f"Successfully processed {result['total_pages']} pages")
        print("\nExtracted Fields:")
        print("-" * 30)
        
        extracted_fields = result["extracted_fields"]
        for field, value in extracted_fields.items():
            print(f"{field.replace('_', ' ').title()}: {value}")
        
        # Save results to JSON file
        output_file = "extracted_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nFull results saved to: {output_file}")

if __name__ == "__main__":
    main()
