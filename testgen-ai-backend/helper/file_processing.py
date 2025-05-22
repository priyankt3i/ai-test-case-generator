# helper/file_processing.py
"""Functions for handling file uploads and extracting text, adapted for FastAPI."""

import io
import json
import pandas as pd
from typing import Optional, Dict, Any, IO

# Attempt to import libraries, handle errors gracefully
try:
    import mammoth
    MAMMOTH_AVAILABLE = True
except ImportError:
    MAMMOTH_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Logging utility
try:
    # Primary attempt: absolute import from project root perspective
    from helper.utils import log_message
except ImportError as e_abs:
    try:
        # Secondary attempt: relative import (if helper is a package and this works)
        from .utils import log_message
    except ImportError as e_rel:
        # Basic fallback logger if all imports fail
        print(f"[ERROR] FAILED TO IMPORT log_message. Abs: {e_abs}. Rel: {e_rel}.")
        def log_message(msg: str, level: str = "INFO", **kwargs):
            print(f"FALLBACK LOGGER [{level}] {msg}")
        log_message("Using fallback logger in file_processing.py.", "WARNING")

def _get_file_content_bytes(file_obj: Any) -> Optional[bytes]:
    """Safely reads bytes from a FastAPI UploadFile or similar file-like object."""
    try:
        # For FastAPI UploadFile, .file is a SpooledTemporaryFile
        if hasattr(file_obj, 'file') and hasattr(file_obj.file, 'read'):
            # Ensure stream is at the beginning if it's seekable
            if callable(getattr(file_obj.file, 'seekable', None)) and file_obj.file.seekable():
                file_obj.file.seek(0)
            return file_obj.file.read()
        # Fallback for other file-like objects that have a read() method directly
        elif hasattr(file_obj, 'read') and callable(file_obj.read):
            if callable(getattr(file_obj, 'seekable', None)) and file_obj.seekable():
                file_obj.seek(0)
            return file_obj.read()
        else:
            log_message("File object does not have a 'file' attribute with a read method, nor a direct 'read' method.", "ERROR")
            return None
    except Exception as e:
        log_message(f"Error reading file content bytes: {e}", "ERROR", exc_info=True)
        return None

# --- DOCX Extraction ---
def extract_text_from_docx(file: Any) -> Optional[str]:
    if not MAMMOTH_AVAILABLE:
        log_message("Cannot extract DOCX: `mammoth` library is missing.", "ERROR")
        return None
    if not file:
        log_message("No file provided for DOCX extraction.", "ERROR")
        return None

    filename = getattr(file, 'filename', 'unknown_file.docx')
    try:
        file_content_bytes = _get_file_content_bytes(file)
        if file_content_bytes is None:
            return None
        
        file_bytes_io = io.BytesIO(file_content_bytes)
        result = mammoth.extract_raw_text(file_bytes_io)
        log_message(f"Successfully extracted text from DOCX: {filename}", "DEBUG")
        return result.value
    except Exception as e:
        log_message(f"Error extracting text from DOCX using mammoth ({filename}): {e}", "ERROR", exc_info=True)
        return None

# --- TXT / MD Extraction ---
def extract_text_from_txt_or_md(file: Any) -> Optional[str]:
    if not file:
        log_message("No file provided for TXT/MD extraction.", "ERROR")
        return None
    
    filename = getattr(file, 'filename', 'unknown_file.txt')
    try:
        file_content_bytes = _get_file_content_bytes(file)
        if file_content_bytes is None:
            return None
        
        try:
            text = file_content_bytes.decode('utf-8')
            log_message(f"Successfully extracted text from TXT/MD (UTF-8): {filename}", "DEBUG")
        except UnicodeDecodeError:
            log_message(f"UTF-8 decoding failed for {filename}. Trying latin-1.", "WARNING")
            text = file_content_bytes.decode('latin-1', errors='replace')
            log_message(f"Successfully extracted text from TXT/MD (latin-1): {filename}", "DEBUG")
        return text
    except Exception as e:
        log_message(f"Error reading/decoding TXT/MD file ({filename}): {e}", "ERROR", exc_info=True)
        return None

# --- XLSX Extraction ---
def extract_text_from_xlsx(file: Any) -> Optional[str]:
    if not file:
        log_message("No file provided for XLSX extraction.", "ERROR")
        return None

    filename = getattr(file, 'filename', 'unknown_file.xlsx')
    try:
        # pd.read_excel can take a file-like object. FastAPI's UploadFile.file is one.
        # Ensure stream is at the beginning
        if hasattr(file, 'file') and callable(getattr(file.file, 'seekable', None)) and file.file.seekable():
            file.file.seek(0)
            
        excel_data = pd.read_excel(file.file, sheet_name=None)
        all_sheets_text = []
        for sheet_name, df in excel_data.items():
            sheet_header = f"--- Sheet: {sheet_name} ---\n"
            sheet_text = df.to_csv(index=False, sep='\t')
            all_sheets_text.append(sheet_header + sheet_text)

        combined_text = "\n\n".join(all_sheets_text)
        log_message(f"Successfully extracted text from XLSX ({len(excel_data)} sheets): {filename}", "DEBUG")
        return combined_text
    except Exception as e:
        log_message(f"Error reading/processing XLSX file ({filename}): {e}", "ERROR", exc_info=True)
        return None

# --- JSON Extraction ---
def extract_text_from_json(file: Any) -> Optional[str]:
    if not file:
        log_message("No file provided for JSON extraction.", "ERROR")
        return None

    filename = getattr(file, 'filename', 'unknown_file.json')
    try:
        file_content_bytes = _get_file_content_bytes(file)
        if file_content_bytes is None:
            return None
            
        try:
            text = file_content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            log_message(f"UTF-8 decoding failed for JSON {filename}. Trying latin-1.", "WARNING")
            text = file_content_bytes.decode('latin-1', errors='replace')

        data = json.loads(text)
        formatted_json = json.dumps(data, indent=2)
        log_message(f"Successfully extracted and formatted JSON: {filename}", "DEBUG")
        return formatted_json
    except json.JSONDecodeError as e:
        log_message(f"Error decoding JSON file ({filename}): {e}", "ERROR")
        return None
    except Exception as e:
        log_message(f"Error reading/processing JSON file ({filename}): {e}", "ERROR", exc_info=True)
        return None

# --- YAML Extraction ---
def extract_text_from_yaml(file: Any) -> Optional[str]:
    if not YAML_AVAILABLE:
         log_message("Cannot extract YAML: `PyYAML` library is missing.", "ERROR")
         return None
    if not file:
        log_message("No file provided for YAML extraction.", "ERROR")
        return None

    filename = getattr(file, 'filename', 'unknown_file.yaml')
    try:
        file_content_bytes = _get_file_content_bytes(file)
        if file_content_bytes is None:
            return None

        try:
            text = file_content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            log_message(f"UTF-8 decoding failed for YAML {filename}. Trying latin-1.", "WARNING")
            text = file_content_bytes.decode('latin-1', errors='replace')

        data = yaml.safe_load(text)
        formatted_yaml = yaml.dump(data, indent=2, allow_unicode=True, sort_keys=False, default_flow_style=False)
        log_message(f"Successfully extracted and formatted YAML: {filename}", "DEBUG")
        return formatted_yaml
    except yaml.YAMLError as e:
        log_message(f"Error parsing YAML file ({filename}): {e}", "ERROR")
        return None
    except Exception as e:
        log_message(f"Error reading/processing YAML file ({filename}): {e}", "ERROR", exc_info=True)
        return None

# --- Generic dispatcher function ---
def extract_text_from_file(file: Any) -> Optional[str]:
     if not file or not hasattr(file, 'filename'): # Basic check for UploadFile-like
         log_message("Invalid file object passed to extract_text_from_file (missing filename).", "ERROR")
         return None

     filename = file.filename
     file_name_lower = filename.lower()

     # Note: _get_file_content_bytes (and by extension, individual extractors)
     # will handle seeking the file stream if necessary.
     # No need to explicitly call file.file.seek(0) here in the dispatcher
     # if the individual extractors correctly use _get_file_content_bytes or manage the stream.

     if file_name_lower.endswith(".docx"):
         return extract_text_from_docx(file)
     elif file_name_lower.endswith((".txt", ".md")):
         return extract_text_from_txt_or_md(file)
     elif file_name_lower.endswith(".xlsx"):
         return extract_text_from_xlsx(file)
     elif file_name_lower.endswith(".json"):
         return extract_text_from_json(file)
     elif file_name_lower.endswith((".yaml", ".yml")):
         return extract_text_from_yaml(file)
     else:
         log_message(f"Unsupported file type for extraction: {filename}", "WARNING")
         return None
