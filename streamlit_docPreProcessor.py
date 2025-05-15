import streamlit as st
import json
import re 
import io 
import os # For environment variables

# Attempt to import necessary libraries
try:
    import fitz  # PyMuPDF
except ImportError:
    st.error("PyMuPDF (fitz) library not found. Please install it: pip install PyMuPDF")
    st.stop()

try:
    from docx import Document
    from docx.shared import Inches 
except ImportError:
    st.error("python-docx library not found. Please install it: pip install python-docx")
    st.stop()

try:
    from PIL import Image
except ImportError:
    st.error("Pillow (PIL) library not found. Please install it: pip install Pillow")
    st.stop()

try:
    import google.generativeai as genai
    from dotenv import load_dotenv
except ImportError:
    st.error("Google Generative AI or python-dotenv library not found. Please install them: pip install google-generativeai python-dotenv")
    st.stop()

# --- Load Environment Variables for Gemini API Key ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in .env file. Please create a .env file with your API key (e.g., GEMINI_API_KEY='AIzaSy...').")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}. Please check your API key and network connection.")


# --- DocumentPreprocessor Class (Remains the same) ---
class DocumentPreprocessor:
    """
    Preprocesses a functional specification document extracted as text for RAG.
    It cleans the text, identifies structural elements, chunks the content,
    and enriches chunks with metadata. Includes points for OCR text integration.
    """
    def __init__(self, raw_text, filename="Uploaded Document"): 
        self.raw_text = raw_text
        self.filename = filename
        self.document_title = "Functional Specification"
        self.document_version = "Unknown"
        self.date_created = ""
        self.date_last_updated = ""
        self.cleaned_lines = []
        self.chunks = []

    def _extract_initial_metadata(self):
        pn_match = re.search(r'PROJECT NAME[^"]*","([^"]+)"', self.raw_text, re.IGNORECASE | re.DOTALL)
        if pn_match:
            project_name_full = pn_match.group(1).strip()
            self.document_title = project_name_full.split('\n')[0].strip() + " Functional Specification"
        else: 
            self.document_title = self.filename.replace('.pdf', '').replace('.txt', '').replace('.docx','').replace('_', ' ').replace('-', ' ')
            self.document_title = ' '.join(word.capitalize() for word in self.document_title.split())

        v_match = re.search(r'"VERSION[^"]*","([^"]+)"', self.raw_text, re.IGNORECASE | re.DOTALL)
        if v_match:
            self.document_version = v_match.group(1).split('\n')[0].strip()
        else:
            version_in_filename = re.search(r'v(\d+\.\d+(?:\.\d+)?)', self.filename, re.IGNORECASE)
            if version_in_filename:
                self.document_version = version_in_filename.group(1)

        dc_match = re.search(r'"Date Created[^"]*","([^"]+)"', self.raw_text, re.IGNORECASE | re.DOTALL)
        if dc_match:
            self.date_created = dc_match.group(1).split('\n')[0].strip()

        du_match = re.search(r'"Date Last Updated[^"]*","([^"]+)"', self.raw_text, re.IGNORECASE | re.DOTALL)
        if du_match:
            self.date_last_updated = du_match.group(1).split('\n')[0].strip()

    def _clean_raw_text(self):
        page_segments = re.split(r"(--- PAGE \d+ ---)", self.raw_text)
        all_final_segments = []
        for i, segment in enumerate(page_segments):
            page_marker_match = re.match(r"--- PAGE (\d+) ---", segment)
            if page_marker_match:
                all_final_segments.append(segment.strip()) 
                continue
            page_text_content = segment
            cleaned_page_text = re.sub(r"Marketing Hub Customer Interface FR\s+Page \d+ of \d+\s+\d{1,2}/\d{1,2}/\d{4}", "", page_text_content, flags=re.IGNORECASE)
            cleaned_page_text = re.sub(r"^\s*a\s*$", "", cleaned_page_text, flags=re.MULTILINE)
            cleaned_page_text = cleaned_page_text.replace('\r\n', '\n')
            initial_lines = cleaned_page_text.split('\n')
            page_final_segments = []
            for line in initial_lines:
                line = line.strip()
                if not line: continue
                segments_on_this_line = []
                if (line.startswith(',"') or line.startswith('"')) and line.endswith('"') and '","' in line:
                    effective_line = line[1:] if line.startswith(',"') else line
                    if effective_line.startswith('"') and effective_line.endswith('"'):
                        core_content = effective_line[1:-1]
                        segments_on_this_line = core_content.split('","')
                    else: segments_on_this_line = [line] 
                elif line.startswith(',"') and line.endswith('"'): segments_on_this_line = [line[2:-1]]
                elif line.startswith('"') and line.endswith('"'): segments_on_this_line = [line[1:-1]]
                else: segments_on_this_line = [line]
                for seg in segments_on_this_line:
                    seg_stripped = seg.strip()
                    if seg_stripped and re.search(r'[a-zA-Z0-9]', seg_stripped):
                        page_final_segments.append(seg_stripped)
            all_final_segments.extend(page_final_segments)
        self.cleaned_lines = all_final_segments

    def _is_section_header(self, line_text):
        if re.match(r"^\d+(\.\d+)*\s+[A-Z0-9].*?(?:\s*\(NOT IN SCOPE\))?$", line_text, re.IGNORECASE):
            if not re.match(r"^\d+(\.\d+)\s+(FR\. No\.|Description|Requirement)", line_text, re.IGNORECASE):
                parts = line_text.split(maxsplit=1) 
                if len(parts) > 1 and parts[1] and parts[1][0].isupper(): return True
        main_section_keywords = [
            "TABLE OF CONTENTS", "VERSION HISTORY", "OVERALL DESCRIPTION", "APPENDIX", 
            "STAKEHOLDER SIGN-OFF AND ACCEPTANCE", "REFERENCE", "LOG IN AND FORGOT PASSWORD", 
            "MARKETING HUB HOME PAGE", "MAIN MENU & SUB MENUS", "DIGITAL PRODUCT PAGE", 
            "SITE NAVIGATION, HEADER AND FOOTER", "PRODUCT SEARCH", "MARKETING BANNERS", 
            "BREADCRUMB TRAIL", "SOCIAL MEDIA", "PHYSICAL PRODUCT PAGE", 
            "CART CHECKOUT FUNCTIONALITY", "MARKETING HUB SECURITY AND USER ACCESS REQUIREMENTS", 
            "SITE ADMINISTRATION FUNCTIONS", "ANALYTICS & REPORTING", "USER SPECIFIC FUNCTIONS", "EMAIL"
        ]
        for kw in main_section_keywords:
            if kw in line_text.upper() and len(line_text) < len(kw) + 25: 
                if "FR. No." not in line_text.upper() and "Requirement" not in line_text.upper():
                    return True
        return False

    def _is_fr_table_header_row(self, line_text):
        return "FR. No." in line_text and "Description" in line_text and "Requirement" in line_text and len(line_text.split()) < 15

    def _is_fr_item_start(self, line_text):
        first_part = line_text.split('","')[0].strip() 
        return bool(re.match(r"^\d+(\.\d+)+$", first_part))

    def _parse_and_chunk(self):
        current_chunk_text_lines = []
        current_section_hierarchy_titles = [] 
        current_fr_number = None
        is_processing_fr_table_content = False 
        current_section_is_out_of_scope = False 
        base_metadata = {
            "document_title": self.document_title, "document_version": self.document_version,
            "date_created": self.date_created, "date_last_updated": self.date_last_updated,
            "source_filename": self.filename
        }
        def finalize_current_chunk():
            nonlocal current_fr_number 
            if not current_chunk_text_lines: return
            chunk_text = "\n".join(current_chunk_text_lines).strip()
            if not chunk_text: 
                current_chunk_text_lines.clear()
                return
            metadata = base_metadata.copy()
            if current_section_hierarchy_titles:
                metadata["current_section_title"] = current_section_hierarchy_titles[-1]
                metadata["section_hierarchy_titles"] = list(current_section_hierarchy_titles[:-1]) if len(current_section_hierarchy_titles) > 1 else []
            else:
                metadata["current_section_title"] = "Document Preamble"; metadata["section_hierarchy_titles"] = []
            if current_fr_number: metadata["fr_number"] = current_fr_number
            metadata["is_out_of_scope"] = current_section_is_out_of_scope
            self.chunks.append({"text": chunk_text, "metadata": metadata})
            current_chunk_text_lines.clear()

        for line_content in self.cleaned_lines:
            if not line_content.strip() or re.fullmatch(r"--- PAGE \d+ ---", line_content.strip()):
                if line_content.strip() and current_chunk_text_lines: 
                     current_chunk_text_lines.append(line_content)
                continue
            is_new_major_section = self._is_section_header(line_content)
            is_fr_header = self._is_fr_table_header_row(line_content)
            is_fr_item = self._is_fr_item_start(line_content) if is_processing_fr_table_content else False
            if is_new_major_section or is_fr_header or is_fr_item:
                finalize_current_chunk()
                if not is_fr_item: current_fr_number = None
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
                    if depth == 1: current_section_hierarchy_titles = [section_title_for_hierarchy]
                    elif depth > len(current_section_hierarchy_titles): current_section_hierarchy_titles.append(section_title_for_hierarchy)
                    else: current_section_hierarchy_titles = current_section_hierarchy_titles[:depth-1] + [section_title_for_hierarchy]
                else: current_section_hierarchy_titles = [section_title_for_hierarchy]
            if is_fr_header: is_processing_fr_table_content = True
            if is_fr_item:
                current_fr_number = line_content.split('","')[0].strip()
                is_processing_fr_table_content = True 
            current_chunk_text_lines.append(line_content)
        finalize_current_chunk() 

    def preprocess(self):
        self._extract_initial_metadata() 
        self._clean_raw_text() 
        self._parse_and_chunk()
        return self.chunks
# --- End of DocumentPreprocessor Class ---


# --- Helper Functions for File Processing & Gemini ---
def extract_text_from_pdf(file_content_bytes):
    full_text = ""
    images_by_page = {} 
    try:
        doc = fitz.open(stream=file_content_bytes, filetype="pdf")
        for page_num_zero_based, page in enumerate(doc):
            page_num_one_based = page_num_zero_based + 1
            full_text += f"--- PAGE {page_num_one_based} ---\n" 
            full_text += page.get_text("text") + "\n"
            img_list = page.get_images(full=True)
            if img_list:
                images_by_page[page_num_one_based] = []
                for img_index, img_info_tuple in enumerate(img_list): 
                    xref = img_info_tuple[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_id = f"P{page_num_one_based}_IMG{img_index+1}"
                    mime_type = f"image/{image_ext.lower()}" if image_ext else "image/unknown"
                    if mime_type == "image/jb2": mime_type = "image/jbig2" 
                    if mime_type == "image/jpg": mime_type = "image/jpeg"
                    images_by_page[page_num_one_based].append({
                        "id": image_id, "data": image_bytes, "ext": image_ext, "mime_type": mime_type
                    })
        doc.close()
        return full_text, images_by_page
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return "Error: Could not extract text from PDF.", {}

def extract_text_from_docx(file_content_bytes):
    full_text = ""
    images_for_docx = [] 
    try:
        document = Document(io.BytesIO(file_content_bytes))
        for para in document.paragraphs:
            full_text += para.text + "\n"
        img_index = 0
        for rel_id in document.part.rels:
            try:
                rel = document.part.rels[rel_id]
                if "image" in rel.target_ref: 
                    image_part = rel.target_part
                    if hasattr(image_part, 'blob'): 
                        image_bytes = image_part.blob
                        image_ext = image_part.content_type.split('/')[-1] 
                        if image_ext == "jpeg": image_ext = "jpg" 
                        image_id = f"DOCX_IMG{img_index+1}"
                        mime_type = image_part.content_type 
                        images_for_docx.append({
                            "id": image_id, "data": image_bytes, "ext": image_ext, "mime_type": mime_type
                        })
                        img_index += 1
            except Exception as e_img_rel:
                st.warning(f"Skipping non-image or problematic relationship in DOCX: {rel.target_ref}, Error: {e_img_rel}")
        return full_text, images_for_docx
    except Exception as e:
        st.error(f"Error processing DOCX: {e}")
        return "Error: Could not extract text from DOCX.", []

# Update cache decorator to include model_name
@st.cache_data
def get_gemini_vision_description(_model_name, image_bytes, image_mime_type, prompt, image_id_for_cache_key):
    """
    Sends image to the specified Gemini model and returns its description.
    _model_name is part of the cache key.
    """
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not configured."
    try:
        # Use the passed model_name
        model = genai.GenerativeModel(_model_name) 
        
        image_part = {
            "mime_type": image_mime_type,
            "data": image_bytes
        }
        
        response = model.generate_content([prompt, image_part])
        
        if response.parts:
            return response.text
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"Error: Content generation blocked due to {response.prompt_feedback.block_reason_message}"
        else:
            return "Error: No content generated by Gemini, or response was empty."

    except Exception as e:
        # st.error(f"Gemini API call failed for model {_model_name} on image {image_id_for_cache_key}: {e}")
        return f"Error calling Gemini API ({_model_name}): {e}"

# --- Streamlit Application ---
st.set_page_config(layout="wide", page_title="RAG Document Preprocessor with AI Vision")
st.title("📄 RAG Document Preprocessor with AI Vision ✨")
st.markdown("""
Upload your document (`.pdf`, `.docx`, or `.txt`). 
For PDFs and DOCX files, extracted images will be shown in the sidebar. 
You can then use Gemini AI to generate descriptions for these images.
""")

# --- File Upload ---
st.sidebar.header("Inputs")
uploaded_file = st.sidebar.file_uploader(
    "1. Upload Document File", 
    type=["pdf", "docx", "txt"]
)

# Initialize/Reset session state variables
if 'filename' not in st.session_state: st.session_state.filename = "Uploaded Document"
if 'raw_text_content' not in st.session_state: st.session_state.raw_text_content = ""
if 'pdf_images' not in st.session_state: st.session_state.pdf_images = {}
if 'docx_images' not in st.session_state: st.session_state.docx_images = []
if 'ai_descriptions' not in st.session_state: st.session_state.ai_descriptions = {}

if uploaded_file:
    st.session_state.filename = uploaded_file.name
    file_content_bytes = uploaded_file.getvalue()
    st.session_state.ai_descriptions = {} 
    st.session_state.pdf_images = {}
    st.session_state.docx_images = []

    if st.session_state.filename.endswith(".pdf"):
        with st.spinner(f"Processing PDF: {st.session_state.filename}..."): # Spinner in main area
            st.session_state.raw_text_content, st.session_state.pdf_images = extract_text_from_pdf(file_content_bytes)
    elif st.session_state.filename.endswith(".docx"):
        with st.spinner(f"Processing DOCX: {st.session_state.filename}..."): # Spinner in main area
            st.session_state.raw_text_content, st.session_state.docx_images = extract_text_from_docx(file_content_bytes)
    elif st.session_state.filename.endswith(".txt"):
        st.session_state.raw_text_content = file_content_bytes.decode("utf-8", errors="replace")
    else:
        st.error("Unsupported file type.")
        st.session_state.raw_text_content = ""

# --- AI Description Section ---
st.sidebar.subheader("2. Generate AI Descriptions for Images")

# LLM Model Selection Dropdown
available_models = ['gemini-1.5-flash-latest', 'gemini-pro-vision', 'gemini-1.5-pro-latest']
selected_model = st.sidebar.selectbox(
    "Choose Gemini Vision Model:",
    available_models,
    index=0 
)

custom_prompt = st.sidebar.text_area(
    "Custom prompt for AI (optional):", 
    "Describe this image, chart, or diagram in detail. If it's a chart or graph, explain its key findings, trends, and data points. If it's a diagram, explain its components and flow. If it's a general image, describe its content and context.",
    height=100
)
has_images_for_ai = False

def display_and_get_ai_description(image_list, source_type, model_name): 
    global has_images_for_ai 
    if image_list:
        has_images_for_ai = True
        st.sidebar.markdown(f"Images found in {source_type}:")
        
        if source_type == "PDF":
            for page_num, images_on_page in sorted(image_list.items()):
                if images_on_page:
                    st.sidebar.markdown(f"**Page {page_num}:**")
                    for img_info in images_on_page:
                        render_image_ai_button(img_info, model_name) 
        else: 
            for img_info in image_list:
                render_image_ai_button(img_info, model_name) 

def render_image_ai_button(img_info, model_name_for_api): 
    """Helper to render image and AI description button/text."""
    try:
        st.sidebar.image(img_info['data'], caption=f"Image: {img_info['id']} ({img_info.get('mime_type', img_info['ext'])})", width=150)
    except Exception as e_img_display:
        st.sidebar.warning(f"Could not display image {img_info['id']}: {e_img_display}")

    button_key = f"gemini_button_{img_info['id']}"
    if st.sidebar.button(f"✨ Get AI Description for {img_info['id']}", key=button_key):
        if not GEMINI_API_KEY:
            st.sidebar.error("Gemini API Key not set. Cannot generate description.")
            st.session_state.ai_descriptions[img_info['id']] = "Error: API Key missing."
        else:
            with st.spinner(f"AI ({model_name_for_api}) is analyzing {img_info['id']}..."): 
                description = get_gemini_vision_description(
                    model_name_for_api, 
                    img_info['data'], 
                    img_info['mime_type'], 
                    custom_prompt,
                    img_info['id'] 
                )
                st.session_state.ai_descriptions[img_info['id']] = "AI Image Description: " + description
    
    if img_info['id'] in st.session_state.ai_descriptions:
        st.sidebar.text_area(
            f"AI Description for {img_info['id']}:", 
            value=st.session_state.ai_descriptions[img_info['id']], 
            height=100, 
            key=f"desc_{img_info['id']}",
            disabled=True 
        )

if st.session_state.pdf_images:
    display_and_get_ai_description(st.session_state.pdf_images, "PDF", selected_model)
if st.session_state.docx_images:
    display_and_get_ai_description(st.session_state.docx_images, "DOCX", selected_model)

if not has_images_for_ai and uploaded_file:
    st.sidebar.info("No images found for AI description, or the format is TXT.")

preprocess_button = st.sidebar.button("🚀 Preprocess Document", type="primary")

# --- Main Area for Output ---
if preprocess_button and st.session_state.raw_text_content:
    integrated_raw_text = st.session_state.raw_text_content
    
    # --- AI Description Integration ---
    if st.session_state.pdf_images:
        temp_lines = integrated_raw_text.splitlines()
        new_text_lines = []
        current_page_for_injection = 0
        for line in temp_lines:
            new_text_lines.append(line)
            page_marker_match = re.match(r"--- PAGE (\d+) ---", line)
            if page_marker_match:
                current_page_for_injection = int(page_marker_match.group(1))
                if current_page_for_injection in st.session_state.pdf_images:
                    for img_info in st.session_state.pdf_images[current_page_for_injection]:
                        ai_desc = st.session_state.ai_descriptions.get(img_info['id'])
                        if ai_desc and not ai_desc.startswith("Error:"):
                            new_text_lines.append(f"\n[AI Description for {img_info['id']} on Page {current_page_for_injection} (Model: {selected_model}) Begin]\n{ai_desc}\n[AI Description for {img_info['id']} End]\n")
        integrated_raw_text = "\n".join(new_text_lines)

    if st.session_state.docx_images:
        docx_ai_appendix = f"\n\n--- AI Descriptions for DOCX Images (Model: {selected_model}) ---\n"
        any_docx_ai_desc = False
        for img_info in st.session_state.docx_images:
            ai_desc = st.session_state.ai_descriptions.get(img_info['id'])
            if ai_desc and not ai_desc.startswith("Error:"):
                docx_ai_appendix += f"[AI Description for {img_info['id']} Begin]\n{ai_desc}\n[AI Description for {img_info['id']} End]\n\n"
                any_docx_ai_desc = True
        if any_docx_ai_desc:
            integrated_raw_text += docx_ai_appendix
    
    # --- Debug: Display Integrated Text ---
    with st.expander("🔍 View Integrated Text (Before Final Chunking)"):
        st.text_area("Text including AI descriptions:", integrated_raw_text, height=400)
    # --- End Debug ---

    with st.spinner("Preprocessing document... Please wait."): 
        try:
            preprocessor = DocumentPreprocessor(
                raw_text=integrated_raw_text, 
                filename=st.session_state.filename
            )
            processed_chunks = preprocessor.preprocess()
            st.header("📊 Preprocessing Results")
            st.markdown(f"**Document Title:** `{preprocessor.document_title}`")
            st.markdown(f"**Document Version:** `{preprocessor.document_version}`")
            st.markdown(f"**Date Created:** `{preprocessor.date_created if preprocessor.date_created else 'N/A'}`")
            st.markdown(f"**Date Last Updated:** `{preprocessor.date_last_updated if preprocessor.date_last_updated else 'N/A'}`")
            st.markdown(f"**Total Chunks Generated:** `{len(processed_chunks)}`")

            if processed_chunks:
                st.subheader("🔍 Processed Chunks Overview")
                num_chunks_to_display = st.slider(
                    "Number of chunks to display:", 
                    min_value=1, max_value=max(1, len(processed_chunks)), 
                    value=min(5, len(processed_chunks)) 
                )
                for i, chunk_data in enumerate(processed_chunks[:num_chunks_to_display]):
                    with st.expander(f"Chunk {i+1}", expanded=False):
                        st.caption(f"Text (first 300 chars):")
                        display_text = chunk_data.get('text', '')
                        st.markdown(f"```\n{display_text[:300]}...\n```" if len(display_text) > 300 else f"```\n{display_text}\n```")
                        st.caption("Metadata:")
                        st.json(chunk_data.get('metadata', {}))
                        if "[AI Description for" in display_text: 
                            st.info("ℹ️ This chunk may contain an AI-generated image description.")
                if len(processed_chunks) > num_chunks_to_display:
                    st.markdown(f"*Displaying first {num_chunks_to_display} of {len(processed_chunks)} chunks.*")
                st.download_button(
                    label="📥 Download All Chunks (JSON)",
                    data=json.dumps(processed_chunks, indent=2),
                    file_name=f"{st.session_state.filename.split('.')[0]}_processed_chunks.json",
                    mime="application/json"
                )
            else:
                st.warning("No chunks were generated. The document might be empty or could not be parsed.")
        except Exception as e:
            st.error(f"An error occurred during preprocessing: {e}")
            st.exception(e) 

elif preprocess_button and not uploaded_file: 
    st.sidebar.warning("Please upload a document file first.")
elif not uploaded_file and not preprocess_button : 
     st.info("Upload a document and click 'Preprocess Document' to begin.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed with the `DocumentPreprocessor` class and Gemini AI.")
st.sidebar.markdown("© 2023 T3innovation. All rights reserved.")
