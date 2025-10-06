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
        # Initialize with DO-only processing enabled
        self.data_parser = DataParser()
        self.data_parser.pdf_extractor.process_only_do_pdfs = True
        self.results = []

    def find_do_pdfs(self, folder_path: str) -> List[str]:
        """Find PDF files that contain 'DO' in their name"""
        pdf_files = []
        folder_path_obj = Path(folder_path)

        if not folder_path_obj.exists():
            return pdf_files

        # Look for PDF files in the folder
        for file in folder_path_obj.glob("*.pdf"):
            filename = file.name

            # Simple check: if "DO" appears anywhere in filename (case insensitive)
            # This catches ALL possible formats without being restrictive
            if 'do' in filename.lower():
                pdf_files.append(str(file))

        return pdf_files

    def get_folder_pdf_summary(self, folder_path: str) -> Dict[str, int]:
        """Get summary of PDFs in folder - total vs DO PDFs"""
        folder_path_obj = Path(folder_path)

        if not folder_path_obj.exists():
            return {"total_pdfs": 0, "do_pdfs": 0, "other_pdfs": 0}

        all_pdfs = list(folder_path_obj.glob("*.pdf"))
        do_pdfs = self.find_do_pdfs(folder_path)

        return {
            "total_pdfs": len(all_pdfs),
            "do_pdfs": len(do_pdfs),
            "other_pdfs": len(all_pdfs) - len(do_pdfs)
        }

    def extract_patient_name_from_folder(self, folder_name: str) -> Dict[str, str]:
        """Extract patient name components from folder name"""
        # Clean up the folder name - remove common prefixes/suffixes
        name = folder_name.strip()

        # Remove common patterns that might not be part of the name
        name = re.sub(r'^(patient|pt|mr|mrs|ms|dr)[\s_-]+', '', name, flags=re.IGNORECASE)
        name = re.sub(r'[\s_-]+(folder|files?)$', '', name, flags=re.IGNORECASE)

        # Replace underscores and dashes with spaces
        name = re.sub(r'[_-]+', ' ', name)

        # Clean up extra spaces
        name = re.sub(r'\s+', ' ', name).strip()

        # Split into name components
        name_parts = name.split() if name else [folder_name]

        if len(name_parts) == 1:
            return {
                "first_name": name_parts[0].title(),
                "middle_name": "",
                "last_name": ""
            }
        elif len(name_parts) == 2:
            return {
                "first_name": name_parts[0].title(),
                "middle_name": "",
                "last_name": name_parts[1].title()
            }
        elif len(name_parts) >= 3:
            return {
                "first_name": name_parts[0].title(),
                "middle_name": name_parts[1].title(),
                "last_name": " ".join(name_parts[2:]).title()
            }

    def parse_physician_name(self, physician_name: str) -> Dict[str, str]:
        """Parse physician name into first, middle, and last name components"""
        if not physician_name:
            return {
                "doctor_first_name": "",
                "doctor_middle_name": "",
                "doctor_last_name": ""
            }

        # Clean the physician name - remove titles and suffixes
        name = physician_name.strip()

        # Remove common titles
        name = re.sub(r'^(dr\.?|doctor|physician)\s+', '', name, flags=re.IGNORECASE)

        # Remove common suffixes at the end
        name = re.sub(r'\s+(md|do|dpm|phd|np|pa)\.?$', '', name, flags=re.IGNORECASE)

        # Clean up extra spaces
        name = re.sub(r'\s+', ' ', name).strip()

        # Split into name parts
        name_parts = name.split() if name else []

        if len(name_parts) == 0:
            return {
                "doctor_first_name": "",
                "doctor_middle_name": "",
                "doctor_last_name": ""
            }
        elif len(name_parts) == 1:
            return {
                "doctor_first_name": name_parts[0].title(),
                "doctor_middle_name": "",
                "doctor_last_name": ""
            }
        elif len(name_parts) == 2:
            # 2 words: first + last
            return {
                "doctor_first_name": name_parts[0].title(),
                "doctor_middle_name": "",
                "doctor_last_name": name_parts[1].title()
            }
        else:
            # 3+ words: first + middle + rest goes to last
            return {
                "doctor_first_name": name_parts[0].title(),
                "doctor_middle_name": name_parts[1].title(),
                "doctor_last_name": " ".join(name_parts[2:]).title()
            }

    def process_patient_folder(self, patient_folder_path: str) -> Dict[str, Any]:
        """Process a single patient folder"""
        folder_path_obj = Path(patient_folder_path)
        patient_name_components = self.extract_patient_name_from_folder(folder_path_obj.name)
        patient_display_name = f"{patient_name_components['first_name']} {patient_name_components['middle_name']} {patient_name_components['last_name']}".strip()

        logger.info(f"Processing patient folder: {patient_display_name}")

        # Get PDF summary for this folder
        pdf_summary = self.get_folder_pdf_summary(patient_folder_path)
        if pdf_summary["other_pdfs"] > 0:
            logger.info(f"  Folder has {pdf_summary['total_pdfs']} PDFs total, {pdf_summary['do_pdfs']} DO PDFs, {pdf_summary['other_pdfs']} other PDFs (skipped)")
        else:
            logger.info(f"  Folder has {pdf_summary['total_pdfs']} PDFs (all are DO PDFs)")

        # Find PDFs with DO in the name
        do_pdfs = self.find_do_pdfs(patient_folder_path)

        if not do_pdfs:
            # No DO PDFs found
            logger.warning(f"No DO PDFs found for patient: {patient_display_name}")
            return {
                "patient_name": patient_display_name,
                "first_name": patient_name_components["first_name"],
                "middle_name": patient_name_components["middle_name"],
                "last_name": patient_name_components["last_name"],
                "folder_path": patient_folder_path,
                "pdf_processed": None,
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
                # Check if it was skipped because it's not a DO PDF
                if result.get("skipped"):
                    logger.info(f"PDF skipped (not a DO file): {pdf_to_process}")
                else:
                    logger.error(f"Error processing PDF {pdf_to_process}: {result['error']}")
                return {
                    "patient_name": patient_display_name,
                    "first_name": patient_name_components["first_name"],
                    "middle_name": patient_name_components["middle_name"],
                    "last_name": patient_name_components["last_name"],
                    "folder_path": patient_folder_path,
                    "pdf_processed": pdf_to_process,
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
                "patient_name": patient_display_name,
                "first_name": patient_name_components["first_name"],
                "middle_name": patient_name_components["middle_name"],
                "last_name": patient_name_components["last_name"],
                "folder_path": patient_folder_path,
                "pdf_processed": pdf_to_process,
                "dob": extracted_fields.get("dob"),
                "address": extracted_fields.get("address"),
                "city": extracted_fields.get("city"),
                "state": extracted_fields.get("state"),
                "postal_code": extracted_fields.get("postal_code"),
                "phone_number": extracted_fields.get("phone_number"),
                "physician_name": extracted_fields.get("physician_name"), # Changed from parsed components to raw string
                "npi": extracted_fields.get("npi"),
                "comment": f"Successfully processed DO PDF ({len(do_pdfs)} DO PDFs found)" if len(do_pdfs) == 1 else f"Successfully processed DO PDF (1 of {len(do_pdfs)} DO PDFs found)"
            }

        except Exception as e:
            logger.error(f"Exception processing PDF {pdf_to_process}: {str(e)}")
            return {
                "patient_name": patient_display_name,
                "first_name": patient_name_components["first_name"],
                "middle_name": patient_name_components["middle_name"],
                "last_name": patient_name_components["last_name"],
                "folder_path": patient_folder_path,
                "pdf_processed": pdf_to_process,
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
                patient_name_components = self.extract_patient_name_from_folder(patient_folder.name)
                patient_display_name = f"{patient_name_components['first_name']} {patient_name_components['middle_name']} {patient_name_components['last_name']}".strip()
                results.append({
                    "patient_name": patient_display_name,
                    "first_name": patient_name_components["first_name"],
                    "middle_name": patient_name_components["middle_name"],
                    "last_name": patient_name_components["last_name"],
                    "folder_path": str(patient_folder),
                    "pdf_processed": None,
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
            "middle_name",
            "last_name",
            "dob",
            "address",
            "city",
            "state",
            "postal_code",
            "phone_number",
            "physician_name", # Changed column name
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
            "Patient Name (Full)",
            "Patient First Name",
            "Patient Middle Name",
            "Patient Last Name",
            "Date of Birth",
            "Address",
            "City",
            "State",
            "Postal Code",
            "Phone Number",
            "Physician Name", # Changed column name
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

    # Default folder path - change this to your actual folder path
    default_folder = r"C:\path\to\your\patient\folders"  # Update this path

    if len(sys.argv) < 2:
        print("Usage: python folder_processor.py <path_to_main_folder> [output_excel_file]")
        print("\nExample:")
        print("python folder_processor.py /path/to/patient/folders patient_report.xlsx")
        print(f"\nOr run without arguments to use default folder: {default_folder}")

        # Use default folder if no arguments provided
        main_folder = default_folder
        output_file = "patient_data_report.xlsx"
    else:
        main_folder = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "patient_data_report.xlsx"

    processor = FolderProcessor()
    processor.run_complete_process(main_folder, output_file)


if __name__ == "__main__":
    main()