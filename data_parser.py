import re
import json
from typing import Dict, List, Optional, Any
from extractor import PDFExtractor


class DataParser:

    def __init__(self):
        self.pdf_extractor = PDFExtractor()

    def extract_field_after_keyword(self,
                                    text: str,
                                    keyword: str,
                                    stop_patterns: List[str] = None,
                                    max_words: int = 5) -> Optional[str]:
        """
        Extract text that appears after a keyword until a stop pattern or max words.
        """
        if stop_patterns is None:
            stop_patterns = [
                r'\n', r'\s{2,}', r'[A-Z][a-z]+[;:]', r'\d{2}/', r'[A-Z]{2,}'
            ]

        # Create pattern to find keyword followed by optional colon/semicolon and spaces
        pattern = rf'{re.escape(keyword)}\s*[;:]?\s*([^\n]*?)(?:' + '|'.join(
            stop_patterns) + ')'

        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            # Limit to max_words
            words = value.split()[:max_words]
            return ' '.join(words) if words else None
        return None

    def extract_first_name(self, text: str) -> Optional[str]:
        """Extract first name from various formats"""
        # Try pattern 1: "First: Name"
        pattern1 = r'First\s*[;:]?\s*([A-Za-z]+)'
        match = re.search(pattern1, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try pattern 2: "Patient Name: First Last"
        pattern2 = r'Patient\s+Name\s*[;:]?\s*([A-Za-z]+)\s+[A-Za-z]+'
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try pattern 3: "Name: First Last" (without "Patient")
        pattern3 = r'(?<!Patient\s)Name\s*[;:]?\s*([A-Za-z]+)\s+[A-Za-z]+'
        match = re.search(pattern3, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return None

    def extract_last_name(self, text: str) -> Optional[str]:
        """Extract last name from various formats"""
        # Try pattern 1: "Last: Name"
        pattern1 = r'Last\s*[;:]?\s*([A-Za-z]+)'
        match = re.search(pattern1, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try pattern 2: "Patient Name: First Last"
        pattern2 = r'Patient\s+Name\s*[;:]?\s*[A-Za-z]+\s+([A-Za-z]+)'
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try pattern 3: "Name: First Last" (without "Patient")
        pattern3 = r'(?<!Patient\s)Name\s*[;:]?\s*[A-Za-z]+\s+([A-Za-z]+)'
        match = re.search(pattern3, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return None

    def extract_dob(self, text: str) -> Optional[str]:
        """Extract date of birth with various formats"""
        # Try multiple DOB patterns
        patterns = [
            r'DOB\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'DOB\s*[;:]\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'Date\s+of\s+Birth\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'dob:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})'  # Handle lowercase "dob:"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def find_patient_info_section(self, text: str) -> Optional[str]:
        """Find the section of text that contains patient information"""
        # Look for patient name patterns first
        patient_patterns = [
            r'Patient\s+Name\s*[;:]?\s*([A-Z][A-Za-z\s]+)',  # "Patient Name: DORIS MOODY"
            r'Patient[;:]?\s*([A-Z][A-Za-z\s]+)',  # "Patient: GLENN NECKERS"
            r'First:\s*[A-Za-z]+\s+Last:\s*[A-Za-z]+',
            r'Name:\s*[A-Za-z]+\s+[A-Za-z]+'
        ]
        
        for pattern in patient_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Found patient name, now find the section around it
                start_pos = max(0, match.start() - 50)
                
                # Look for end of patient section - be more specific about physician markers
                end_markers = [
                    r'NPI\s*[;:]?\s*\d{10}',  # Physician NPI (10 digits)
                    r'Physician\s+Signature',
                    r'ordering\s+Physician',
                    r'By\s+signing\s+above',
                    r'Date\s+of\s+Encounter',
                    r'Length\s+of\s+need',
                    r'HCPCS\s+CODE',
                    # Look for a second "Address:" that might be physician's
                    r'Address:\s*\d+[^A-Z]*[A-Z]{2}\s+\d{5}.*?Address:\s*\d+'
                ]
                
                end_pos = len(text)
                search_text = text[match.start():]
                
                for end_pattern in end_markers:
                    end_match = re.search(end_pattern, search_text, re.IGNORECASE)
                    if end_match:
                        end_pos = match.start() + end_match.start()
                        break
                
                patient_section = text[start_pos:end_pos]
                
                # Additional check: if we find multiple addresses, keep only up to the first complete address
                address_matches = list(re.finditer(r'Address:\s*(\d+[^\n]*)', patient_section, re.IGNORECASE))
                if len(address_matches) > 1:
                    # Keep text only up to after the first address
                    first_address_end = address_matches[0].end()
                    # Look for next major section (like NPI, codes, etc.)
                    remaining_text = patient_section[first_address_end:]
                    section_break = re.search(r'(NPI|HCPCS|Length of need)', remaining_text, re.IGNORECASE)
                    if section_break:
                        end_pos = start_pos + first_address_end + section_break.start()
                        patient_section = text[start_pos:end_pos]
                
                return patient_section
        
        return None

    def extract_first_address(self,
                              text: str,
                              patient_context: str = None) -> Optional[str]:
        """Extract patient address from patient info section"""
        # First, try to find the patient information section
        patient_section = self.find_patient_info_section(text)
        
        if patient_section:
            # Look for address in patient section - be more restrictive
            # Look for "My Address:" first (patient-specific)
            patient_address_pattern = r'My\s+Address\s*[;:]?\s*(\d+[^\n]*?)(?=\s*(?:Address:|City:|State:|Postal|Phone|ITEM|HCPCS|NPI|Physician|$|\n))'
            match = re.search(patient_address_pattern, patient_section, re.IGNORECASE)
            
            if match:
                address = match.group(1).strip()
                # Clean and validate - remove any trailing "Address:" pattern
                address = re.sub(r'\s+Address:\s*.*$', '', address, flags=re.IGNORECASE)
                address = re.sub(r'\s+', ' ', address)
                if self.is_valid_address(address):
                    return address
            
            # If no "My Address", look for first "Address:" in patient section
            general_address_pattern = r'Address\s*[;:]?\s*(\d+[^\n]*?)(?=\s*(?:Address:|City:|State:|Postal|Phone|ITEM|HCPCS|NPI|Physician|$|\n))'
            matches = list(re.finditer(general_address_pattern, patient_section, re.IGNORECASE))
            
            if matches:
                # Take only the first address found in patient section
                match = matches[0]
                address = match.group(1).strip()
                # Clean and validate - remove any trailing "Address:" pattern
                address = re.sub(r'\s+Address:\s*.*$', '', address, flags=re.IGNORECASE)
                address = re.sub(r'\s+', ' ', address)
                if self.is_valid_address(address):
                    return address
        
        # Extended address extraction patterns - try these if patient section doesn't work
        extended_patterns = [
            # Original "My Address" pattern
            r'My\s+Address\s*[;:]?\s*(\d+[^\n]*?)(?=\s*(?:Address:|City:|State:|Phone|ITEM|HCPCS|NPI|Physician|$|\n))',
            
            # Standard "Address:" patterns
            r'Address\s*[;:]?\s*(\d+[^\n]*?)(?=\s*(?:Address:|City:|State:|Phone|Patient|Primary|Postal|DOB|Physician|NPI|ITEM|HCPCS|$|\n))',
            
            # Patient Address patterns
            r'Patient\s+Address\s*[;:]?\s*(\d+[^\n]*?)(?=\s*(?:Address:|City:|State:|Phone|Patient|Primary|Postal|DOB|Physician|NPI|$|\n))',
            
            # Street address patterns (common formats)
            r'(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Drive|Dr\.?|Lane|Ln\.?|Boulevard|Blvd\.?)\s*[;:]?\s*(\d+[^\n]*?)(?=\s*(?:Address:|City:|State:|Phone|Patient|Primary|Postal|DOB|Physician|NPI|$|\n))',
            
            # Numeric address at start of line
            r'^(\d+\s+[A-Za-z][^\n]*?)(?=\s*(?:Address:|City:|State:|Phone|Patient|Primary|Postal|DOB|Physician|NPI|$|\n))',
            
            # Address with apartment/unit numbers
            r'Address\s*[;:]?\s*(\d+[^\n]*?(?:Apt|Suite|Unit|#)\s*[A-Za-z0-9]*[^\n]*?)(?=\s*(?:Address:|City:|State:|Phone|Patient|Primary|Postal|DOB|Physician|NPI|$|\n))',
            
            # PO Box patterns
            r'(?:P\.?O\.?\s*Box|Post\s+Office\s+Box)\s*[;:]?\s*(\d+[^\n]*?)(?=\s*(?:Address:|City:|State:|Phone|Patient|Primary|Postal|DOB|Physician|NPI|$|\n))',
            
            # Generic address line patterns
            r'(?:Home\s+)?Address\s*[;:]?\s*(\d+[^\n]*?)(?=\s*(?:Address:|City:|State:|Phone|Patient|Primary|Postal|DOB|Physician|NPI|ITEM|HCPCS|$|\n))',
            
            # Address in context of patient info
            r'(?:Patient|Client|Member)\s+(?:Home\s+)?Address\s*[;:]?\s*(\d+[^\n]*?)(?=\s*(?:Address:|City:|State:|Phone|Patient|Primary|Postal|DOB|Physician|NPI|$|\n))'
        ]
        
        for pattern in extended_patterns:
            if patient_section:
                # Try pattern in patient section first
                match = re.search(pattern, patient_section, re.IGNORECASE | re.MULTILINE)
                if match:
                    address = match.group(1).strip()
                    # Clean any trailing "Address:" patterns
                    address = re.sub(r'\s+Address:\s*.*$', '', address, flags=re.IGNORECASE)
                    address = re.sub(r'\s+', ' ', address)
                    if self.is_valid_address(address):
                        return address
            
            # Try pattern in full text
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                address = match.group(1).strip()
                # Clean any trailing "Address:" patterns
                address = re.sub(r'\s+Address:\s*.*$', '', address, flags=re.IGNORECASE)
                address = re.sub(r'\s+', ' ', address)
                if self.is_valid_address(address):
                    return address

        return None

    def is_valid_address(self, address: str) -> bool:
        """Validate if an address looks like a real address"""
        if not address or len(address) > 150:
            return False
        
        # Clean the address
        address_clean = address.strip()
        
        # Must start with a number or PO Box
        if not re.match(r'^(\d+|P\.?O\.?\s*Box)', address_clean, re.IGNORECASE):
            return False
        
        # Must have some letters (street name)
        if not re.search(r'[A-Za-z]', address_clean):
            return False
            
        # Should not contain medical terms
        medical_terms = [
            'orthosis', 'knee', 'medical', 'treatment', 'diagnosis',
            'necessity', 'prescribed', 'elevation', 'medication', 'hcpcs',
            'code', 'item', 'description', 'brace', 'device', 'dmepos',
            'supplier', 'authorization', 'certification', 'length of need'
        ]
        
        address_lower = address_clean.lower()
        if any(term in address_lower for term in medical_terms):
            return False
        
        # Should look like a real address (has common address components)
        address_indicators = [
            r'\b(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd|way|place|pl|court|ct|circle|cir)\b',
            r'\b(?:apt|apartment|suite|unit|#)\s*[a-z0-9]+',
            r'^\d+\s+[a-z]',  # Number followed by letter (street name)
            r'p\.?o\.?\s*box\s*\d+',  # PO Box
        ]
        
        has_address_indicator = any(re.search(pattern, address_lower) for pattern in address_indicators)
        
        # Basic length check - real addresses are usually reasonable length
        word_count = len(address_clean.split())
        
        return has_address_indicator and 2 <= word_count <= 10

    def extract_city(self, text: str) -> Optional[str]:
        """Extract city after 'City:' keyword - get the first occurrence"""
        # Look for City with colon or semicolon, get first match
        pattern = r'City\s*[;:]\s*([A-Za-z\s]+?)(?=\s*(?:City|State|Address|Phone|Patient|Primary|Postal|DOB|Physician|$))'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            # Return the first city found, cleaned up
            city = match.group(1).strip()
            # Remove extra whitespace and limit words
            words = city.split()[:3]  # Max 3 words for city name
            # Remove any trailing punctuation or weird characters
            clean_words = []
            for word in words:
                clean_word = re.sub(r'[^A-Za-z\s]', '', word).strip()
                if clean_word and not any(
                        med_word in clean_word.lower()
                        for med_word in ['orthosis', 'medical', 'treatment']):
                    clean_words.append(clean_word)
            return ' '.join(clean_words) if clean_words else None
        return None

    def extract_state(self, text: str) -> Optional[str]:
        """Extract state after 'State:' keyword - get first occurrence from context"""
        # Look for state pattern (2 letter codes typically) - get first match
        # More specific pattern to avoid picking up random 2-letter combinations
        pattern = r'State\s*[;:]?\s*([A-Z]{2})(?=\s|$|\n)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

    def extract_postal_code(self, text: str) -> Optional[str]:
        """Extract postal code from patient info section first"""
        # First, try to find the patient information section
        patient_section = self.find_patient_info_section(text)
        
        if patient_section:
            # Look for postal code in patient section first
            patterns = [
                r'Zip\s*[;:]?\s*(\d{5}(?:-\d{4})?)',  # "Zip: 29053"
                r'Postal\s+Code\s*[;:]?\s*(\d{5}(?:-\d{4})?)',
                r'Postal\s+[Cc][ao]de?\s*[;:]?\s*(\d{5}(?:-\d{4})?)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, patient_section, re.IGNORECASE)
                if match:
                    return match.group(1)
        
        # Fallback: search entire text line by line
        lines = text.split('\n')
        for line in lines:
            # Skip lines that contain medical necessity text
            if any(med_word in line.lower() for med_word in
                ['orthosis', 'medical necessity', 'treatment', 'prescribed']):
                continue

            # Try "Zip:" first
            pattern_zip = r'Zip\s*[;:]?\s*(\d{5}(?:-\d{4})?)'
            match = re.search(pattern_zip, line, re.IGNORECASE)
            if match:
                return match.group(1)

            # Try exact match first in this line
            pattern1 = r'Postal\s+Code\s*[;:]?\s*(\d{5}(?:-\d{4})?)'
            match = re.search(pattern1, line, re.IGNORECASE)
            if match:
                return match.group(1)

            # Try with common typos like "Postal Cade", "Postal Code", etc.
            pattern2 = r'Postal\s+[Cc][ao]de?\s*[;:]?\s*(\d{5}(?:-\d{4})?)'
            match = re.search(pattern2, line, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def extract_phone_number(self, text: str) -> Optional[str]:
        """Extract patient phone number"""
        # Try multiple phone number patterns
        patterns = [
            r'Patient\s+Phone\s+Number\s*:?\s*(\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})',
            r'Patient\s+Phone\s*:?\s*(\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})',
            r'Phone\s*:?\s*(\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})(?=\s*(?:Primary|Private|Fax|$|\n))'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def extract_physician_name(self, text: str) -> Optional[str]:
        """Extract physician name from various formats"""
        patterns = [
            r'Ordering\s+Physician\s*[;:]?\s*([A-Za-z\s.]+?)(?=\s*(?:MD|DO|DPM|$|\n|NPI|Address))',
            r'Physician\s+Name\s*[;:]?\s*([A-Za-z\s.]+?)(?=\s*(?:MD|DO|DPM|$|\n|NPI|Address))',
            r'Physician\s*[;:]?\s*([A-Za-z\s.]+?)(?=\s*(?:MD|DO|DPM|$|\n|NPI|Address))'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up common suffixes that might be captured
                name = re.sub(r'\s*(MD|DO|DPM)\s*$', '', name, flags=re.IGNORECASE)
                return name.strip() if name else None
        return None

    def extract_npi(self, text: str) -> Optional[str]:
        """Extract NPI (National Provider Identifier) - 10 digit number"""
        patterns = [
            r'NPI\s*[;:]?\s*(\d{10})',
            r'National\s+Provider\s+Identifier\s*[;:]?\s*(\d{10})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    

    

    

    def parse_document_data(self, documents: List[Any]) -> Dict[str, Any]:
        """
        Parse extracted documents and extract structured data
        """
        # Combine all page content
        full_text = ""
        for doc in documents:
            if hasattr(doc, 'page_content'):
                full_text += doc.page_content + "\n"
            elif isinstance(doc, dict) and 'Content' in doc:
                full_text += doc['Content'] + "\n"

        # Extract patient context first to help with address extraction
        patient_name_context = None
        patient_name_match = re.search(
            r'(?:Patient Name|First:|Last:).*(?:\n)', full_text, re.IGNORECASE)
        if patient_name_match:
            patient_name_context = patient_name_match.group(0)

        dob_context = None
        dob_match = re.search(r'DOB\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
                              full_text, re.IGNORECASE)
        if dob_match:
            # Try to capture a small window around DOB
            start_index = max(0, dob_match.start() - 50)
            end_index = min(len(full_text), dob_match.end() + 50)
            dob_context = full_text[start_index:end_index]

        # Combine context for address extraction
        combined_patient_context = ""
        if patient_name_context:
            combined_patient_context += patient_name_context
        if dob_context:
            combined_patient_context += dob_context

        # Extract all fields
        extracted_data = {
            "first_name":
            self.extract_first_name(full_text),
            "last_name":
            self.extract_last_name(full_text),
            "dob":
            self.extract_dob(full_text),
            "address":
            self.extract_first_address(full_text, combined_patient_context),
            "city":
            self.extract_city(full_text),
            "state":
            self.extract_state(full_text),
            "postal_code":
            self.extract_postal_code(full_text),
            "phone_number":
            self.extract_phone_number(full_text),
            "physician_name":
            self.extract_physician_name(full_text),
            "npi":
            self.extract_npi(full_text)
        }

        return extracted_data

    def process_pdf_file(self,
                         pdf_path: str,
                         debug_dir: str = None) -> Dict[str, Any]:
        """
        Complete pipeline: Extract PDF text and parse structured data
        """
        try:
            # Extract documents from PDF
            documents = self.pdf_extractor.text_extractor(pdf_path, debug_dir)

            # Parse structured data
            structured_data = self.parse_document_data(documents)

            # Add metadata
            result = {
                "pdf_path": pdf_path,
                "total_pages": len(documents),
                "extracted_fields": structured_data,
                "raw_documents": documents
            }

            return result

        except Exception as e:
            return {
                "pdf_path": pdf_path,
                "error": str(e),
                "extracted_fields": None,
                "raw_documents": None
            }


# Example usage and testing
if __name__ == "__main__":
    parser = DataParser()

    # Test with sample data from the real examples
    sample_data1 = [{
        "Content":
        "athena 09-13-2024 11:04 AM ET , 613-243925974 pa Zof2 IC PA« Pp . DONAHOO, Wyatt (id #853, dob: 10/09/1946) & Sig mrgtton AUTHORIZATION PRESCRIPTION REQUEST FORM FOR HIP ORTHOSIS i Healthcare please send RX Form & Pertinent Chart Notes Fax No: 709-704-2065 Ciimneeting Aootbh, eotadinns PLEASE SEND THIS FORM BACK IN 3 BUSINESS DAYS Date:09/00/2024 First: Wyatt Last: Donahoo Physician Name: L.J. Patrick Bell D.O. DOB: 10/09/1946 NPI: 1871589309 Address: 5549 Phillips 103 Rd Address: 626 Poplar St City; Poplar Grove City: Helena State: AR State: AR Postal Code: 72374 Patient Phone Number: 8708166435 Primary Ins: Medicare — Policy #: SG44UJ4EP64"
    }]

    sample_data2 = [{
        "Content":
        "PLEA SO Pio PO DATA DN OS BUSINESS DATS Date: 08/27/2014 Patient Name: Doald Hagon Physician Name; DR Brian M Lott DOB; 03/31/1940 NPI; 1801862826 Address: 19014 Auburn Rd Address: 150 N Wnut St | City: Chagrin Fas City: Chillicthe State: OH State: OH Postal Cade: 44023 Postal code: 45601 Patient Phone Number: 4405434942 Phone Number: 7407794500"
    }]

    sample_data3 = [{
        "Content":
        "Sep/11/2024 1:48:30 PM OSF Healthcare 8155381359 2i2 Atin: Michael Lewis PRIOR AUTHORIZATION REQUEST FORM FOR HIP ORTHOSIS Fax 700-300-4802 Ph; 786-565-3816 Please Send RX Form & Pertinent Chart Notes Fax No: 709-300-1892 Sunshine Madical Supplies PLEASE SEND THIS FORM BACK IN 3 BUSINESS DAYS Date: 08/26/2024 First; Darlene Last: Piecha Physician Name: Dr, Dexter Angeles MD DOB: 09/06/1935 NPI: 1629335781 Address: 422 W Ist St Address: 1436 Midtown Rd City: Oglesby City: Peru State; IL State: IL Postal Code: 61348 Postal code: 61354 Patient Phone Number: 8158839232 Phone Number: 8155381355 Primary Ins: Medicare Policy #: SVM6GHTOUTS2| Fax Number: 8155451359 Private Ins: Policy #: Height: Weight: This patievt is being treated under a comprehensive plan af care for Hip pain."
    }]

    sample_data4 = [{
        "Content":
        "09/20/2024 FRI 12:43 FAX 44054392684 B001/002 HIPAA DME Medical Record Report *RE: Medical Necessity Certification for [Donald Hagen]* Name: Donald Hagen DOB: 03/31/1940 Ordering Physician: DR Brian M Lott MD Diagnosis: M23.51, M2352 Back Brace (L0650) Hip Brace (L1686) Knee Brace (L1851) Ankle Brace (L1971) Shoulder Brace (L3960) Elbow Brace (L3761) Wrist Brace (L3916)"
    }]

    print("Testing Sample 1 (First/Last format):")
    result1 = parser.parse_document_data(sample_data1)
    print(json.dumps(result1, indent=2))

    print("\nTesting Sample 2 (Patient Name format):")
    result2 = parser.parse_document_data(sample_data2)
    print(json.dumps(result2, indent=2))

    print("\nTesting Sample 3 (Contextual Address):")
    result3 = parser.parse_document_data(sample_data3)
    print(json.dumps(result3, indent=2))

    print("\nTesting Sample 4 (Donald Hagen format with Ordering Physician):")
    result4 = parser.parse_document_data(sample_data4)
    print(json.dumps(result4, indent=2))

    # Example of processing a PDF file (uncomment to use)
    # pdf_path = "path/to/your/pdf/file.pdf"
    # result = parser.process_pdf_file(pdf_path)
    # print(json.dumps(result, indent=2, default=str))
