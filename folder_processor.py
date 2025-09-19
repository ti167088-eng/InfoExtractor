
import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from data_parser import DataParser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FolderProcessor:
    def __init__(self):
        self.data_parser = DataParser()
        self.results = []
    
    def find_do_pdfs(self, folder_path: str) -> List[str]:
        """Find PDF files that contain 'DO' in their name"""
        pdf_files = []
        folder_path_obj = Path(folder_path)
        
        if not folder_path_obj.exists():
            return pdf_files
            
        # Look for PDF files in the folder
        for file in folder_path_obj.glob("*.pdf"):
            filename = file.name.lower()
            # Check if filename contains 'do' as a word (not just as part of another word)
            if re.search(r'\bdo\b', filename, re.IGNORECASE):
                pdf_files.append(str(file))
        
        return pdf_files
    
    def extract_patient_name_from_folder(self, folder_name: str) -> str:
        """Extract patient name from folder name"""
        # Clean up the folder name - remove common prefixes/suffixes
        name = folder_name.strip()
        
        # Remove common patterns that might not be part of the name
        name = re.sub(r'^(patient|pt|mr|mrs|ms|dr)[\s_-]+', '', name, flags=re.IGNORECASE)
        name = re.sub(r'[\s_-]+(folder|files?)$', '', name, flags=re.IGNORECASE)
        
        # Replace underscores and dashes with spaces
        name = re.sub(r'[_-]+', ' ', name)
        
        # Clean up extra spaces
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name.title() if name else folder_name
    
    def process_patient_folder(self, patient_folder_path: str) -> Dict[str, Any]:
        """Process a single patient folder"""
        folder_path_obj = Path(patient_folder_path)
        patient_name = self.extract_patient_name_from_folder(folder_path_obj.name)
        
        logger.info(f"Processing patient folder: {patient_name}")
        
        # Find PDFs with DO in the name
        do_pdfs = self.find_do_pdfs(patient_folder_path)
        
        if not do_pdfs:
            # No DO PDFs found
            logger.warning(f"No DO PDFs found for patient: {patient_name}")
            return {
                "patient_name": patient_name,
                "folder_path": patient_folder_path,
                "pdf_processed": None,
                "first_name": None,
                "last_name": None,
                "dob": None,
                "address": None,
                "city": None,
                "state": None,
                "postal_code": None,
                "phone_number": None,
                "physician_name": None,
                "npi": None,
                "comment": "DO file not found"
            }
        
        # Process the first DO PDF found
        pdf_to_process = do_pdfs[0]
        logger.info(f"Processing PDF: {pdf_to_process}")
        
        try:
            # Use our data parser to extract information
            result = self.data_parser.process_pdf_file(pdf_to_process)
            
            if result.get("error"):
                logger.error(f"Error processing PDF {pdf_to_process}: {result['error']}")
                return {
                    "patient_name": patient_name,
                    "folder_path": patient_folder_path,
                    "pdf_processed": pdf_to_process,
                    "first_name": None,
                    "last_name": None,
                    "dob": None,
                    "address": None,
                    "city": None,
                    "state": None,
                    "postal_code": None,
                    "phone_number": None,
                    "physician_name": None,
                    "npi": None,
                    "comment": f"Error processing PDF: {result['error']}"
                }
            
            extracted_fields = result.get("extracted_fields", {})
            
            # Combine the patient name from folder with extracted data
            return {
                "patient_name": patient_name,
                "folder_path": patient_folder_path,
                "pdf_processed": pdf_to_process,
                "first_name": extracted_fields.get("first_name"),
                "last_name": extracted_fields.get("last_name"),
                "dob": extracted_fields.get("dob"),
                "address": extracted_fields.get("address"),
                "city": extracted_fields.get("city"),
                "state": extracted_fields.get("state"),
                "postal_code": extracted_fields.get("postal_code"),
                "phone_number": extracted_fields.get("phone_number"),
                "physician_name": extracted_fields.get("physician_name"),
                "npi": extracted_fields.get("npi"),
                "comment": f"Successfully processed DO PDF ({len(do_pdfs)} DO PDFs found)" if len(do_pdfs) == 1 else f"Successfully processed DO PDF (1 of {len(do_pdfs)} DO PDFs found)"
            }
            
        except Exception as e:
            logger.error(f"Exception processing PDF {pdf_to_process}: {str(e)}")
            return {
                "patient_name": patient_name,
                "folder_path": patient_folder_path,
                "pdf_processed": pdf_to_process,
                "first_name": None,
                "last_name": None,
                "dob": None,
                "address": None,
                "city": None,
                "state": None,
                "postal_code": None,
                "phone_number": None,
                "physician_name": None,
                "npi": None,
                "comment": f"Exception processing PDF: {str(e)}"
            }
    
    def process_main_folder(self, main_folder_path: str) -> List[Dict[str, Any]]:
        """Process the main folder containing patient subfolders"""
        main_path = Path(main_folder_path)
        
        if not main_path.exists():
            raise FileNotFoundError(f"Main folder not found: {main_folder_path}")
        
        results = []
        
        # Get all subdirectories (patient folders)
        patient_folders = [f for f in main_path.iterdir() if f.is_dir()]
        
        if not patient_folders:
            logger.warning(f"No patient folders found in: {main_folder_path}")
            return results
        
        logger.info(f"Found {len(patient_folders)} patient folders")
        
        # Process each patient folder
        for patient_folder in patient_folders:
            try:
                result = self.process_patient_folder(str(patient_folder))
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing patient folder {patient_folder}: {str(e)}")
                patient_name = self.extract_patient_name_from_folder(patient_folder.name)
                results.append({
                    "patient_name": patient_name,
                    "folder_path": str(patient_folder),
                    "pdf_processed": None,
                    "first_name": None,
                    "last_name": None,
                    "dob": None,
                    "address": None,
                    "city": None,
                    "state": None,
                    "postal_code": None,
                    "phone_number": None,
                    "physician_name": None,
                    "npi": None,
                    "comment": f"Error processing folder: {str(e)}"
                })
        
        return results
    
    def create_excel_report(self, results: List[Dict[str, Any]], output_file: str = "patient_data_report.xlsx"):
        """Create an Excel file with all the results"""
        if not results:
            logger.warning("No results to write to Excel")
            return
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Reorder columns for better readability
        column_order = [
            "patient_name",
            "first_name", 
            "last_name",
            "dob",
            "address",
            "city",
            "state",
            "postal_code",
            "phone_number",
            "physician_name",
            "npi",
            "pdf_processed",
            "folder_path",
            "comment"
        ]
        
        # Ensure all columns exist
        for col in column_order:
            if col not in df.columns:
                df[col] = None
        
        df = df[column_order]
        
        # Rename columns for better display
        df.columns = [
            "Patient Name (from folder)",
            "First Name (from PDF)",
            "Last Name (from PDF)",
            "Date of Birth",
            "Address",
            "City",
            "State",
            "Postal Code",
            "Phone Number",
            "Physician Name",
            "NPI",
            "PDF Processed",
            "Folder Path",
            "Comments"
        ]
        
        try:
            # Write to Excel
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Patient Data', index=False)
                
                # Get the workbook and worksheet to format
                workbook = writer.book
                worksheet = writer.sheets['Patient Data']
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            logger.info(f"Excel report created: {output_file}")
            print(f"Excel report saved as: {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating Excel file: {str(e)}")
            # Fallback: save as CSV
            csv_file = output_file.replace('.xlsx', '.csv')
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved as CSV instead: {csv_file}")
            print(f"Excel creation failed, saved as CSV: {csv_file}")
    
    def run_complete_process(self, main_folder_path: str, output_file: str = "patient_data_report.xlsx"):
        """Run the complete process from folder scanning to Excel creation"""
        logger.info(f"Starting complete folder processing for: {main_folder_path}")
        
        try:
            # Process all patient folders
            results = self.process_main_folder(main_folder_path)
            
            if not results:
                print("No patient folders processed successfully.")
                return
            
            # Create Excel report
            self.create_excel_report(results, output_file)
            
            # Print summary
            total_patients = len(results)
            successful_extractions = len([r for r in results if r.get("comment", "").startswith("Successfully")])
            do_files_not_found = len([r for r in results if r.get("comment") == "DO file not found"])
            errors = total_patients - successful_extractions - do_files_not_found
            
            print(f"\n{'='*50}")
            print(f"PROCESSING SUMMARY")
            print(f"{'='*50}")
            print(f"Total patient folders processed: {total_patients}")
            print(f"Successfully extracted data: {successful_extractions}")
            print(f"DO files not found: {do_files_not_found}")
            print(f"Errors encountered: {errors}")
            print(f"Output file: {output_file}")
            print(f"{'='*50}")
            
        except Exception as e:
            logger.error(f"Error in complete process: {str(e)}")
            print(f"Error: {str(e)}")


def main():
    """Main function for command line usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python folder_processor.py <path_to_main_folder> [output_excel_file]")
        print("\nExample:")
        print("python folder_processor.py /path/to/patient/folders patient_report.xlsx")
        return
    
    main_folder = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "patient_data_report.xlsx"
    
    processor = FolderProcessor()
    processor.run_complete_process(main_folder, output_file)


if __name__ == "__main__":
    main()
