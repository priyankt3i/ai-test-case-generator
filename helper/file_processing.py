# helper/file_processing.py
"""Functions for handling file uploads and extracting text."""

import io
import json
import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import Optional, Dict, Any

# Attempt to import libraries, handle errors gracefully
try:
    import mammoth
    MAMMOTH_AVAILABLE = True
except ImportError:
    MAMMOTH_AVAILABLE = False
    # Warning logged during docx extraction attempt if needed

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    # Warning logged during pdf extraction attempt if needed

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    # Warning logged during yaml extraction attempt if needed

# Pandas is used for Excel, error handled within the function

# Assuming utils.py is accessible for logging
try:
    from .utils import log_message # Use relative import if utils is in the same 'helper' folder
except ImportError:
     # Fallback if relative import fails or utils is elsewhere
     try:
          from utils import log_message
     except ImportError:
          # Basic fallback logger
          def log_message(msg, level, **kwargs): print(f"[{level}] {msg}")
          log_message("Could not import log_message from utils in file_processing.", "WARNING")


# --- DOCX Extraction ---
def extract_text_from_docx(file: UploadedFile) -> Optional[str]:
    """
    Extracts raw text content from an uploaded .docx file using mammoth.

    Args:
        file: The Streamlit UploadedFile object (.docx).

    Returns:
        The extracted text as a string, or None if extraction fails or
        mammoth is not available.
    """
    if not MAMMOTH_AVAILABLE:
        log_message("Cannot extract DOCX: `mammoth` library is missing. Install with `pip install python-mammoth`", "ERROR")
        # st.error("Cannot extract text from .docx: `mammoth` library is missing.") # Avoid direct st calls here
        return None

    if not file:
        log_message("No file provided for DOCX extraction.", "ERROR")
        return None

    try:
        file_bytes = io.BytesIO(file.getvalue())
        result = mammoth.extract_raw_text(file_bytes)
        log_message(f"Successfully extracted text from DOCX: {file.name}", "DEBUG")
        return result.value
    except AttributeError as e:
        log_message(f"Invalid file object provided for DOCX extraction ({file.name}): {e}", "ERROR", exc_info=True)
        return None
    except Exception as e:
        log_message(f"Error extracting text from DOCX using mammoth ({file.name}): {e}", "ERROR", exc_info=True)
        return None

# --- PDF Extraction ---
def extract_text_from_pdf(file: UploadedFile) -> Optional[str]:
    """
    Extracts text content from an uploaded .pdf file using PyMuPDF (fitz).

    Args:
        file: The Streamlit UploadedFile object (.pdf).

    Returns:
        The extracted text as a string, or None if extraction fails or
        PyMuPDF is not available.
    """
    if not FITZ_AVAILABLE:
        log_message("Cannot extract PDF: `PyMuPDF` (fitz) library is missing. Install with `pip install PyMuPDF`", "ERROR")
        return None

    if not file:
        log_message("No file provided for PDF extraction.", "ERROR")
        return None

    try:
        file_bytes = file.getvalue()
        full_text = ""
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                full_text += page.get_text()
        log_message(f"Successfully extracted text from PDF: {file.name}", "DEBUG")
        return full_text
    except Exception as e:
        log_message(f"Error extracting text from PDF using PyMuPDF ({file.name}): {e}", "ERROR", exc_info=True)
        return None

# --- TXT / MD Extraction ---
def extract_text_from_txt_or_md(file: UploadedFile) -> Optional[str]:
    """
    Extracts text content from an uploaded .txt or .md file.
    Attempts decoding with UTF-8, then latin-1 as a fallback.

    Args:
        file: The Streamlit UploadedFile object (.txt or .md).

    Returns:
        The extracted text as a string, or None if extraction fails.
    """
    if not file:
        log_message("No file provided for TXT/MD extraction.", "ERROR")
        return None

    try:
        file_bytes = file.getvalue()
        try:
            text = file_bytes.decode('utf-8')
            log_message(f"Successfully extracted text from TXT/MD (UTF-8): {file.name}", "DEBUG")
        except UnicodeDecodeError:
            log_message(f"UTF-8 decoding failed for {file.name}. Trying latin-1.", "WARNING")
            text = file_bytes.decode('latin-1', errors='replace') # Fallback encoding
            log_message(f"Successfully extracted text from TXT/MD (latin-1): {file.name}", "DEBUG")
        return text
    except Exception as e:
        log_message(f"Error reading/decoding TXT/MD file ({file.name}): {e}", "ERROR", exc_info=True)
        return None

# --- XLSX Extraction ---
def extract_text_from_xlsx(file: UploadedFile) -> Optional[str]:
    """
    Extracts text content from all sheets of an uploaded .xlsx file using pandas.
    Converts each sheet to a CSV-like string representation.

    Args:
        file: The Streamlit UploadedFile object (.xlsx).

    Returns:
        A string containing the content of all sheets, or None if extraction fails.
    """
    if not file:
        log_message("No file provided for XLSX extraction.", "ERROR")
        return None

    try:
        # Read all sheets into a dictionary of DataFrames
        excel_data = pd.read_excel(file, sheet_name=None)
        all_sheets_text = []
        for sheet_name, df in excel_data.items():
            # Add sheet name as header
            sheet_header = f"--- Sheet: {sheet_name} ---\n"
            # Convert DataFrame to CSV string format (or use df.to_string(), df.to_markdown())
            # Using CSV representation here for simplicity
            sheet_text = df.to_csv(index=False, sep='\t') # Use tab separation for less comma collision
            all_sheets_text.append(sheet_header + sheet_text)

        combined_text = "\n\n".join(all_sheets_text)
        log_message(f"Successfully extracted text from XLSX ({len(excel_data)} sheets): {file.name}", "DEBUG")
        return combined_text
    except Exception as e:
        log_message(f"Error reading/processing XLSX file ({file.name}): {e}", "ERROR", exc_info=True)
        # st.warning(f"Could not read Excel file '{file.name}'. Ensure it's a valid .xlsx file.") # Avoid direct st calls
        return None

# --- JSON Extraction ---
def extract_text_from_json(file: UploadedFile) -> Optional[str]:
    """
    Extracts and formats text content from an uploaded .json file.

    Args:
        file: The Streamlit UploadedFile object (.json).

    Returns:
        A formatted JSON string, or None if extraction/parsing fails.
    """
    if not file:
        log_message("No file provided for JSON extraction.", "ERROR")
        return None

    try:
        file_bytes = file.getvalue()
        try:
            text = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            log_message(f"UTF-8 decoding failed for JSON {file.name}. Trying latin-1.", "WARNING")
            text = file_bytes.decode('latin-1', errors='replace')

        data = json.loads(text)
        # Return a nicely formatted string representation
        formatted_json = json.dumps(data, indent=2)
        log_message(f"Successfully extracted and formatted JSON: {file.name}", "DEBUG")
        return formatted_json
    except json.JSONDecodeError as e:
        log_message(f"Error decoding JSON file ({file.name}): {e}", "ERROR")
        return None # Indicates parsing failure
    except Exception as e:
        log_message(f"Error reading/processing JSON file ({file.name}): {e}", "ERROR", exc_info=True)
        return None

# --- YAML Extraction ---
def extract_text_from_yaml(file: UploadedFile) -> Optional[str]:
    """
    Extracts and formats text content from an uploaded .yaml or .yml file.

    Args:
        file: The Streamlit UploadedFile object (.yaml or .yml).

    Returns:
        A formatted YAML string, or None if extraction/parsing fails or PyYAML is missing.
    """
    if not YAML_AVAILABLE:
         log_message("Cannot extract YAML: `PyYAML` library is missing. Install with `pip install PyYAML`", "ERROR")
         return None

    if not file:
        log_message("No file provided for YAML extraction.", "ERROR")
        return None

    try:
        file_bytes = file.getvalue()
        try:
            text = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            log_message(f"UTF-8 decoding failed for YAML {file.name}. Trying latin-1.", "WARNING")
            text = file_bytes.decode('latin-1', errors='replace')

        data = yaml.safe_load(text)
        # Return a nicely formatted string representation
        formatted_yaml = yaml.dump(data, indent=2, allow_unicode=True, sort_keys=False, default_flow_style=False)
        log_message(f"Successfully extracted and formatted YAML: {file.name}", "DEBUG")
        return formatted_yaml
    except yaml.YAMLError as e:
        log_message(f"Error parsing YAML file ({file.name}): {e}", "ERROR")
        return None # Indicates parsing failure
    except Exception as e:
        log_message(f"Error reading/processing YAML file ({file.name}): {e}", "ERROR", exc_info=True)
        return None

# --- You might also want a generic dispatcher function ---
def extract_text_from_file(file: UploadedFile) -> Optional[str]:
     """
     Detects file type and calls the appropriate text extraction function.

     Args:
         file: The Streamlit UploadedFile object.

     Returns:
         The extracted text as a string, or None if extraction fails or type is unsupported.
     """
     if not file:
         return None

     file_name_lower = file.name.lower()

     if file_name_lower.endswith(".docx"):
         return extract_text_from_docx(file)
     elif file_name_lower.endswith(".pdf"):
         return extract_text_from_pdf(file)
     elif file_name_lower.endswith((".txt", ".md")):
         return extract_text_from_txt_or_md(file)
     elif file_name_lower.endswith(".xlsx"):
         return extract_text_from_xlsx(file)
     elif file_name_lower.endswith(".json"):
         return extract_text_from_json(file)
     elif file_name_lower.endswith((".yaml", ".yml")):
         return extract_text_from_yaml(file)
     else:
         log_message(f"Unsupported file type for extraction: {file.name}", "WARNING")
         # Optionally try reading as plain text as a last resort?
         # return extract_text_from_txt_or_md(file)
         return None
