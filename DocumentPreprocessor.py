import re
import json

class DocumentPreprocessor:
    """
    Preprocesses a functional specification document extracted as text for RAG.
    It cleans the text, identifies structural elements, chunks the content,
    and enriches chunks with metadata. Includes points for OCR text integration.
    """
    def __init__(self, raw_text, filename="Marketing Hub Customer Interface FR v3.0.pdf", ocr_texts_by_page=None):
        """
        Initializes the preprocessor.

        Args:
            raw_text (str): The full raw text content of the document.
            filename (str): The original filename, used for fallback metadata.
            ocr_texts_by_page (dict, optional): A dictionary mapping page numbers (int)
                                              to OCR-extracted text from images on those pages.
                                              Example: {1: "Text from image on page 1", ...}
                                              The user is responsible for populating this dict using
                                              external OCR tools.
        """
        self.raw_text = raw_text
        self.filename = filename
        self.ocr_texts_by_page = ocr_texts_by_page if ocr_texts_by_page else {} # OCR INTEGRATION
        self.document_title = "Functional Specification" # Default
        self.document_version = "Unknown" # Default
        self.date_created = ""
        self.date_last_updated = ""
        self.cleaned_lines = [] # Stores processed lines/segments from the raw text
        self.chunks = [] # Stores the final list of {'text': ..., 'metadata': ...}

    def _extract_initial_metadata(self):
        """
        Extracts document-level metadata (title, version, dates) from the raw text.
        Uses heuristics based on observed patterns in the document.
        """
        # Project Name / Document Title
        pn_match = re.search(r'PROJECT NAME[^"]*","([^"]+)"', self.raw_text, re.IGNORECASE | re.DOTALL)
        if pn_match:
            project_name_full = pn_match.group(1).strip()
            self.document_title = project_name_full.split('\n')[0].strip() + " Functional Specification"
        else: 
            self.document_title = self.filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
            self.document_title = ' '.join(word.capitalize() for word in self.document_title.split())

        # Version
        v_match = re.search(r'"VERSION[^"]*","([^"]+)"', self.raw_text, re.IGNORECASE | re.DOTALL)
        if v_match:
            self.document_version = v_match.group(1).split('\n')[0].strip()
        else:
            version_in_filename = re.search(r'v(\d+\.\d+(?:\.\d+)?)', self.filename, re.IGNORECASE)
            if version_in_filename:
                self.document_version = version_in_filename.group(1)

        # Date Created
        dc_match = re.search(r'"Date Created[^"]*","([^"]+)"', self.raw_text, re.IGNORECASE | re.DOTALL)
        if dc_match:
            self.date_created = dc_match.group(1).split('\n')[0].strip()

        # Date Last Updated
        du_match = re.search(r'"Date Last Updated[^"]*","([^"]+)"', self.raw_text, re.IGNORECASE | re.DOTALL)
        if du_match:
            self.date_last_updated = du_match.group(1).split('\n')[0].strip()

    def _clean_raw_text(self):
        """
        Cleans the raw text. It processes the document page by page,
        applies cleaning rules, integrates OCR text if provided for that page,
        and then parses the content into logical segments.
        """
        # OCR INTEGRATION: Split document into pages first to associate OCR text
        # The pattern looks for "--- PAGE " followed by digits and " ---"
        # Using re.split to keep the delimiters (page markers) for page number extraction,
        # but we'll filter them out later or handle them.
        # A more robust way is to find all matches and process text between them.
        
        page_segments = re.split(r"(--- PAGE \d+ ---)", self.raw_text)
        
        all_final_segments = []
        current_page_number = 0 # Start with 0 or 1 depending on how pages are numbered

        for i, segment in enumerate(page_segments):
            page_marker_match = re.match(r"--- PAGE (\d+) ---", segment)
            if page_marker_match:
                current_page_number = int(page_marker_match.group(1))
                # This segment is the marker itself, skip adding its text directly
                continue
            
            # This segment is the actual text content for a page (or text before the first page marker)
            page_text_content = segment

            # 1. Remove page footers and other major noise patterns from this page's text
            cleaned_page_text = re.sub(r"Marketing Hub Customer Interface FR\s+Page \d+ of \d+\s+\d{1,2}/\d{1,2}/\d{4}", "", page_text_content, flags=re.IGNORECASE)
            cleaned_page_text = re.sub(r"^\s*a\s*$", "", cleaned_page_text, flags=re.MULTILINE) # Remove standalone 'a' lines

            # 2. Normalize line endings to \n
            cleaned_page_text = cleaned_page_text.replace('\r\n', '\n')

            # OCR INTEGRATION: Append OCR text for the current page if available
            if current_page_number in self.ocr_texts_by_page:
                ocr_text_for_page = self.ocr_texts_by_page[current_page_number]
                # Add a separator or integrate more intelligently if needed
                cleaned_page_text += "\n\n[OCR Text Begin]\n" + ocr_text_for_page + "\n[OCR Text End]\n\n"

            # 3. Split the cleaned page text into initial lines and process them to extract logical segments
            initial_lines = cleaned_page_text.split('\n')
            page_final_segments = []

            for line in initial_lines:
                line = line.strip()
                if not line: # Skip empty lines
                    continue

                segments_on_this_line = []
                if (line.startswith(',"') or line.startswith('"')) and line.endswith('"') and '","' in line:
                    effective_line = line
                    if line.startswith(',"'): 
                        effective_line = line[1:] 
                    if effective_line.startswith('"') and effective_line.endswith('"'):
                        core_content = effective_line[1:-1]
                        segments_on_this_line = core_content.split('","')
                    else: 
                        segments_on_this_line = [line] 
                elif line.startswith(',"') and line.endswith('"'): 
                    segments_on_this_line = [line[2:-1]]
                elif line.startswith('"') and line.endswith('"'): 
                    segments_on_this_line = [line[1:-1]]
                else: 
                    segments_on_this_line = [line]
                
                for seg in segments_on_this_line:
                    seg_stripped = seg.strip()
                    if seg_stripped and len(seg_stripped) > 0 and re.search(r'[a-zA-Z0-9]', seg_stripped):
                        page_final_segments.append(seg_stripped)
            
            all_final_segments.extend(page_final_segments)
        
        self.cleaned_lines = all_final_segments


    def _is_section_header(self, line_text):
        """Checks if a line is a main section header (e.g., "1. Section", "1.1 Subsection")."""
        if re.match(r"^\d+(\.\d+)*\s+[A-Z0-9].*?(?:\s*\(NOT IN SCOPE\))?$", line_text, re.IGNORECASE):
            if not re.match(r"^\d+(\.\d+)\s+(FR\. No\.|Description|Requirement)", line_text, re.IGNORECASE):
                parts = line_text.split(maxsplit=1) 
                if len(parts) > 1 and parts[1] and parts[1][0].isupper():
                    return True
        
        main_section_keywords = [
            "TABLE OF CONTENTS", "VERSION HISTORY", "OVERALL DESCRIPTION", 
            "APPENDIX", "STAKEHOLDER SIGN-OFF AND ACCEPTANCE", "REFERENCE",
            "LOG IN AND FORGOT PASSWORD", "MARKETING HUB HOME PAGE", "MAIN MENU & SUB MENUS",
            "DIGITAL PRODUCT PAGE", "SITE NAVIGATION, HEADER AND FOOTER", "PRODUCT SEARCH",
            "MARKETING BANNERS", "BREADCRUMB TRAIL", "SOCIAL MEDIA", "PHYSICAL PRODUCT PAGE",
            "CART CHECKOUT FUNCTIONALITY", "MARKETING HUB SECURITY AND USER ACCESS REQUIREMENTS",
            "SITE ADMINISTRATION FUNCTIONS", "ANALYTICS & REPORTING", "USER SPECIFIC FUNCTIONS", "EMAIL"
        ]
        for kw in main_section_keywords:
            if kw in line_text.upper() and len(line_text) < len(kw) + 25: 
                if "FR. No." not in line_text.upper() and "Requirement" not in line_text.upper():
                    return True
        return False

    def _is_fr_table_header_row(self, line_text):
        """Checks if a line is the header row of a Functional Requirement table."""
        return "FR. No." in line_text and "Description" in line_text and "Requirement" in line_text and len(line_text.split()) < 15


    def _is_fr_item_start(self, line_text):
        """
        Checks if a line signals the start of a new Functional Requirement item.
        """
        first_part = line_text.split('","')[0].strip() 
        if re.match(r"^\d+(\.\d+)+$", first_part): 
            return True
        return False

    def _parse_and_chunk(self):
        """
        Parses the cleaned lines to identify structure and create chunks.
        Each chunk is a dictionary with 'text' and 'metadata'.
        """
        current_chunk_text_lines = []
        current_section_hierarchy_titles = [] 
        current_fr_number = None
        is_processing_fr_table_content = False 
        current_section_is_out_of_scope = False 

        self._extract_initial_metadata() # Ensure document metadata is available
        base_metadata = {
            "document_title": self.document_title,
            "document_version": self.document_version,
            "date_created": self.date_created,
            "date_last_updated": self.date_last_updated,
            "source_filename": self.filename
        }

        def finalize_current_chunk():
            nonlocal current_fr_number 
            if not current_chunk_text_lines:
                return

            chunk_text = "\n".join(current_chunk_text_lines).strip()
            if not chunk_text: 
                current_chunk_text_lines.clear()
                return

            metadata = base_metadata.copy()
            
            if current_section_hierarchy_titles:
                metadata["current_section_title"] = current_section_hierarchy_titles[-1]
                if len(current_section_hierarchy_titles) > 1:
                    metadata["section_hierarchy_titles"] = list(current_section_hierarchy_titles[:-1])
                else:
                    metadata["section_hierarchy_titles"] = [] 
            else:
                metadata["current_section_title"] = "Document Preamble" 
                metadata["section_hierarchy_titles"] = []

            if current_fr_number:
                metadata["fr_number"] = current_fr_number
            
            metadata["is_out_of_scope"] = current_section_is_out_of_scope
            
            self.chunks.append({"text": chunk_text, "metadata": metadata})
            current_chunk_text_lines.clear()

        for line_content in self.cleaned_lines:
            is_new_major_section = self._is_section_header(line_content)
            is_fr_header = self._is_fr_table_header_row(line_content)
            is_fr_item = self._is_fr_item_start(line_content) if is_processing_fr_table_content else False
            
            should_finalize_previous_chunk = False
            if is_new_major_section or is_fr_header or is_fr_item:
                should_finalize_previous_chunk = True

            if should_finalize_previous_chunk:
                finalize_current_chunk()
                if not is_fr_item : 
                    current_fr_number = None

            if is_new_major_section:
                is_processing_fr_table_content = False 
                current_section_is_out_of_scope = "NOT IN SCOPE" in line_content.upper()
                
                match = re.match(r"^(\d+(?:\.\d+)*)\s*(.*)", line_content)
                section_title_for_hierarchy = line_content.strip() 
                if match:
                    num_part, title_part = match.groups()
                    title_part_cleaned = re.sub(r"\s*\(NOT IN SCOPE\).*$", "", title_part.strip(), flags=re.IGNORECASE).strip()
                    section_title_for_hierarchy = title_part_cleaned
                    
                    depth = num_part.count('.') + 1
                    if depth == 1: 
                        current_section_hierarchy_titles = [section_title_for_hierarchy]
                    elif depth > len(current_section_hierarchy_titles): 
                        current_section_hierarchy_titles.append(section_title_for_hierarchy)
                    else: 
                        current_section_hierarchy_titles = current_section_hierarchy_titles[:depth-1] + [section_title_for_hierarchy]
                else: 
                    current_section_hierarchy_titles = [section_title_for_hierarchy]
            
            if is_fr_header:
                is_processing_fr_table_content = True

            if is_fr_item:
                current_fr_number = line_content.split('","')[0].strip()
                is_processing_fr_table_content = True 

            current_chunk_text_lines.append(line_content)

        finalize_current_chunk() 

    def preprocess(self):
        """
        Executes the full preprocessing workflow.
        """
        self._extract_initial_metadata() 
        self._clean_raw_text() # This now handles OCR text integration if data is provided
        self._parse_and_chunk()
        return self.chunks

    def get_chunks_as_json(self, indent=2):
        """Returns the processed chunks as a JSON string."""
        return json.dumps(self.chunks, indent=indent)

# Example Usage (assuming 'full_text_content' is the string from content_fetcher)
if __name__ == '__main__':
    full_text_content = """
--- PAGE 1 ---

a


,"FUNCTIONAL SPECIFICATION
"
"PROJECT NAME:
 PROJECT CODE
","Marketing Hub


NMC-24-0016
"
--- PAGE 2 ---
TABLE OF CONTENTS

"VERSION HISTORY
","3
"
--- PAGE 8 ---
This is some text on page 8.
Imagine an image here. [source: 31] More text after image.
    """

    # --- This is where you would get the actual full text from your file ---
    # full_text_content = actual_pdf_extracted_text
    
    # OCR INTEGRATION EXAMPLE:
    # User would populate this dictionary using their own OCR tools/process.
    # For example, if an image on page 8 contained important diagram text:
    ocr_data_for_pdf = {
        1: "This is OCR text from an image on page 1. For example, a logo description.",
        8: "Diagram: User clicks button -> System validates input -> Confirmation message shown."
    }
    # If no images or no OCR text, pass None or an empty dict.

    print("Simulating preprocessing with a sample of the document content and OCR data.")
    print("For full processing, replace 'full_text_content' with the actual document text.\n")
    
    preprocessor = DocumentPreprocessor(
        raw_text=full_text_content, 
        filename="Marketing Hub Customer Interface FR v3.0.pdf",
        ocr_texts_by_page=ocr_data_for_pdf # OCR INTEGRATION
    )
    processed_chunks = preprocessor.preprocess()

    print(f"Successfully processed into {len(processed_chunks)} chunks.")
    print("Document Title:", preprocessor.document_title)
    print("Document Version:", preprocessor.document_version)
    
    for i, chunk in enumerate(processed_chunks): 
        print(f"\n--- Chunk {i+1} ---")
        print("Text:", chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text']) 
        print("Metadata:", chunk['metadata'])
        if "[OCR Text Begin]" in chunk['text']:
            print(">>> This chunk contains integrated OCR text. <<<")

    # To save to a file:
    # with open("processed_chunks_with_ocr.json", "w") as f:
    #     f.write(preprocessor.get_chunks_as_json())
    # print("\nSaved all chunks to processed_chunks_with_ocr.json")

