"""Functions for handling file uploads and extracting text."""

import io
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Attempt to import mammoth, handle error gracefully
try:
    import mammoth
    MAMMOTH_AVAILABLE = True
except ImportError:
    MAMMOTH_AVAILABLE = False
    st.error("`mammoth` library not found. Please install it (`pip install python-mammoth`) to process .docx files.")


def extract_text_from_docx(file: UploadedFile) -> str | None:
    """
    Extracts raw text content from an uploaded .docx file using mammoth.

    Args:
        file: The Streamlit UploadedFile object (.docx).

    Returns:
        The extracted text as a string, or None if extraction fails or
        mammoth is not available.
    """
    if not MAMMOTH_AVAILABLE:
        st.error("Cannot extract text: `mammoth` library is missing.")
        return None

    if not file:
        st.error("No file provided for text extraction.")
        return None

    try:
        # Read file content into BytesIO stream
        file_bytes = io.BytesIO(file.getvalue())
        # Extract raw text
        result = mammoth.extract_raw_text(file_bytes)
        return result.value
    except AttributeError:
        # Might happen if file object is not as expected
        st.error("Invalid file object provided.")
        return None
    except Exception as e:
        # Catch potential errors during mammoth processing
        st.error(f"Error extracting text from DOCX using mammoth: {e}")
        return None
