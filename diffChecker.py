import streamlit as st
import difflib
import html
import io
import time # Import time for potential benchmarking if needed later

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Advanced Side-by-Side Diff Tool")

# --- Constants ---
# Define allowed file types for upload
ALLOWED_FILE_TYPES = [
    "txt", "py", "js", "html", "css", "md", "java", "c", "cpp", "h",
    "cs", "go", "rs", "php", "rb", "swift", "kt", "kts", "yaml", "json"
]

# --- Styling ---
# Inject custom CSS for highlighting and layout
st.markdown("""
<style>
    /* General Layout */
    .diff-container {
        display: flex; /* <<< Key style for side-by-side */
        width: 100%;
        font-family: 'Courier New', Courier, monospace; /* Strict monospace */
        font-size: 14px;       /* Slightly larger font */
        line-height: 1.5;      /* Increased line height */
        border: 1px solid #ced4da; /* Slightly darker border */
        border-radius: 0.25rem;
        overflow: hidden;
        background-color: #ffffff; /* Ensure white background */
    }
    .diff-pane {
        width: 50%; /* <<< Key style for side-by-side */
        overflow: auto; /* Allow both horizontal and vertical scroll */
        padding: 1rem;
        border-right: 1px solid #dee2e6; /* Clearer separator line */
        max-height: 600px; /* Limit height and make pane scrollable */
        box-sizing: border-box; /* Include padding and border in the element's total width and height */
    }
    .diff-pane:last-child {
        border-right: none;
    }
    .diff-pane pre {
        margin: 0;
        padding: 0;
        white-space: pre;    /* Keep whitespace, disable wrapping */
        word-wrap: normal;   /* Disable word wrapping */
        background-color: transparent; /* Inherit background */
        border: none;        /* Remove any default pre border */
        font-family: inherit;/* Inherit font from parent */
        font-size: inherit;  /* Inherit font size */
    }

    /* Line Numbering */
    .line-num {
        display: inline-block;
        width: 45px; /* Slightly wider */
        text-align: right;
        color: #6c757d; /* Gray color */
        padding-right: 10px;
        user-select: none; /* Prevent selecting line numbers */
        border-right: 1px solid #e9ecef; /* Subtle line number separator */
        margin-right: 10px; /* More space after number */
        font-size: 12px; /* Smaller line numbers */
    }

    /* Line Highlighting Classes */
    .diff-line {
        display: block; /* Ensure each line takes full width */
        min-height: 1.5em; /* Ensure consistent line height */
        border-radius: 2px; /* Subtle rounding */
        margin-bottom: 1px; /* Tiny space between lines */
    }
    .diff-equal { background-color: #ffffff; } /* White */
    .diff-insert { background-color: #d1f7d8; } /* Softer green */
    .diff-delete { background-color: #fde2e4; } /* Softer red */
    .diff-replace { background-color: #fff3cd; } /* Softer yellow */
    .diff-empty {
        background-color: #f8f9fa; /* Light gray */
        min-height: 1.5em;
    }
    .diff-empty .line-num, .diff-empty span:not(.line-num) {
        color: #f8f9fa; /* Make text invisible */
        user-select: none;
    }

    /* Intra-line Highlighting Classes */
    .char-insert {
        background-color: #a3e9b3; /* Stronger green */
        font-weight: bold;
        border-radius: 2px;
        padding: 0 1px; /* Add slight padding */
    }
    .char-delete {
        background-color: #f7c1c6; /* Stronger red */
        text-decoration: line-through;
        border-radius: 2px;
        padding: 0 1px; /* Add slight padding */
    }

    /* Ensure input text areas are also monospace */
    div[data-testid="stTextArea"] textarea {
        font-family: 'Courier New', Courier, monospace;
        font-size: 14px;
        line-height: 1.5;
        border-color: #ced4da;
    }

    /* Style file uploader */
    div[data-testid="stFileUploader"] { margin-bottom: 0.5rem; }

    /* Style buttons */
    div[data-testid="stButton"] button { margin-top: 1rem; border-radius: 0.25rem; }

    /* Visual whitespace characters */
    .ws-space { color: #ced4da; } /* Light gray for space dots */
    .ws-tab { color: #adb5bd; } /* Slightly darker gray for tab arrows */

</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def visualize_whitespace(escaped_text):
    """Replaces spaces and tabs with visible characters IN ALREADY ESCAPED TEXT."""
    vis_text = escaped_text.replace(' ', '<span class="ws-space">·</span>')
    vis_text = vis_text.replace('\t', '<span class="ws-tab">→</span>&nbsp;&nbsp;&nbsp;')
    return vis_text

def generate_intra_line_diff(line_left, line_right, show_whitespace=False):
    """Performs character-level diff on two lines and returns HTML with highlights."""
    # This function remains the same as before
    html_left, html_right = [], []
    if line_left is None and line_right is None: return "&nbsp;", "&nbsp;"
    if line_left is None:
        escaped_right = html.escape(line_right)
        processed_right = visualize_whitespace(escaped_right) if show_whitespace else escaped_right
        return "&nbsp;", f'<span class="char-insert">{processed_right}</span>' if processed_right else "&nbsp;"
    if line_right is None:
        escaped_left = html.escape(line_left)
        processed_left = visualize_whitespace(escaped_left) if show_whitespace else escaped_left
        return f'<span class="char-delete">{processed_left}</span>' if processed_left else "&nbsp;", "&nbsp;"

    char_matcher = difflib.SequenceMatcher(None, line_left, line_right)
    for tag, i1, i2, j1, j2 in char_matcher.get_opcodes():
        seg_left, seg_right = line_left[i1:i2], line_right[j1:j2]
        escaped_left, escaped_right = html.escape(seg_left), html.escape(seg_right)
        processed_left = visualize_whitespace(escaped_left) if show_whitespace else escaped_left
        processed_right = visualize_whitespace(escaped_right) if show_whitespace else escaped_right

        if tag == 'equal':
            html_left.append(processed_left)
            html_right.append(processed_right)
        elif tag == 'replace':
            html_left.append(f'<span class="char-delete">{processed_left}</span>' if processed_left else '')
            html_right.append(f'<span class="char-insert">{processed_right}</span>' if processed_right else '')
        elif tag == 'delete':
            html_left.append(f'<span class="char-delete">{processed_left}</span>' if processed_left else '')
        elif tag == 'insert':
            html_right.append(f'<span class="char-insert">{processed_right}</span>' if processed_right else '')

    final_html_left = "".join(html_left)
    final_html_right = "".join(html_right)
    return final_html_left if final_html_left else "&nbsp;", final_html_right if final_html_right else "&nbsp;"

# --- MODIFICATION: Added enable_intra_line_diff parameter ---
def get_diff_lines(text1, text2, ignore_whitespace=False, case_insensitive=False, show_whitespace=False, enable_intra_line_diff=True):
    """Compares texts line-by-line and optionally char-by-char, returning lists for rendering."""
    lines1, lines2 = text1.splitlines(), text2.splitlines()
    processed_lines1, processed_lines2 = [], []
    def preprocess_line(line):
        proc_line = line
        if ignore_whitespace: proc_line = proc_line.strip()
        if case_insensitive: proc_line = proc_line.lower()
        return proc_line
    processed_lines1 = [preprocess_line(line) for line in lines1]
    processed_lines2 = [preprocess_line(line) for line in lines2]

    line_matcher = difflib.SequenceMatcher(None, processed_lines1, processed_lines2, autojunk=False)
    output_left, output_right = [], []
    line_num_left, line_num_right = 1, 1

    for tag, i1, i2, j1, j2 in line_matcher.get_opcodes():
        lines_left_orig, lines_right_orig = lines1[i1:i2], lines2[j1:j2]
        len_left, len_right = len(lines_left_orig), len(lines_right_orig)
        max_len = max(len_left, len_right)

        for i in range(max_len):
            line_left = lines_left_orig[i] if i < len_left else None
            line_right = lines_right_orig[i] if i < len_right else None
            num_l = line_num_left + i if line_left is not None else None
            num_r = line_num_right + i if line_right is not None else None

            # --- MODIFICATION: Conditional intra-line diff ---
            if tag == 'replace' and enable_intra_line_diff:
                # Perform detailed intra-line diff (potentially slow)
                content_l, content_r = generate_intra_line_diff(line_left, line_right, show_whitespace)
                css_l = 'diff-replace' if line_left is not None else 'diff-empty'
                css_r = 'diff-replace' if line_right is not None else 'diff-empty'
                output_left.append((num_l, content_l, css_l))
                output_right.append((num_r, content_r, css_r))
            # --- END MODIFICATION ---
            else:
                # Handle equal, insert, delete, or replace (when intra-line is disabled)
                content_l, content_r = "&nbsp;", "&nbsp;"
                css_l, css_r = 'diff-empty', 'diff-empty' # Default for placeholders

                if tag == 'equal':
                    css_l, css_r = 'diff-equal', 'diff-equal'
                elif tag == 'replace': # No intra-line diff, just highlight line
                    css_l = 'diff-replace' if line_left is not None else 'diff-empty'
                    css_r = 'diff-replace' if line_right is not None else 'diff-empty'
                elif tag == 'delete':
                    css_l = 'diff-delete'
                    css_r = 'diff-empty' # Placeholder on right
                elif tag == 'insert':
                    css_l = 'diff-empty' # Placeholder on left
                    css_r = 'diff-insert'

                # Format content for non-intra-line cases
                if line_left is not None:
                    escaped_l = html.escape(line_left)
                    content_l = visualize_whitespace(escaped_l) if show_whitespace else escaped_l
                    content_l = content_l if content_l.strip() else "&nbsp;"
                if line_right is not None:
                    escaped_r = html.escape(line_right)
                    content_r = visualize_whitespace(escaped_r) if show_whitespace else escaped_r
                    content_r = content_r if content_r.strip() else "&nbsp;"

                output_left.append((num_l, content_l, css_l))
                output_right.append((num_r, content_r, css_r))


        line_num_left += len_left
        line_num_right += len_right
    return output_left, output_right

def render_diff_pane(lines):
    """Generates HTML string for a single diff pane."""
    # This function remains the same
    html_lines = []
    for line_num, content_html, css_class in lines:
        line_num_str = str(line_num) if line_num is not None else "&nbsp;"
        display_content = content_html if content_html else "&nbsp;"
        html_lines.append(
            f'<div class="diff-line {css_class}">'
            f'<span class="line-num">{line_num_str}</span>'
            f'<span>{display_content}</span>'
            f'</div>'
        )
    return "<pre>" + "\n".join(html_lines) + "</pre>"

def read_uploaded_file(uploaded_file):
    """Reads content from an uploaded file."""
    # This function remains the same
    if uploaded_file is not None:
        try:
            stringio = io.TextIOWrapper(uploaded_file, encoding='utf-8', errors='replace')
            content = stringio.read()
            uploaded_file.seek(0)
            return content
        except Exception as e:
            st.error(f"Error reading file '{uploaded_file.name}': {e}")
            return ""
    return ""

# --- Streamlit App UI ---

st.title("✨ Advanced Side-by-Side Diff Tool")
st.markdown("Upload files or paste text/code to see line-by-line and intra-line differences highlighted.")

# --- Session State Initialization ---
if 'text1' not in st.session_state: st.session_state.text1 = ""
if 'text2' not in st.session_state: st.session_state.text2 = ""
if 'current_file1_file_id' not in st.session_state: st.session_state.current_file1_file_id = None
if 'current_file2_file_id' not in st.session_state: st.session_state.current_file2_file_id = None

# --- Input Areas ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original (Left)")
    uploaded_file1 = st.file_uploader("Upload File 1", type=ALLOWED_FILE_TYPES, key="uploader1", help="Upload the first file")
    if uploaded_file1 is not None:
        if st.session_state.current_file1_file_id != uploaded_file1.file_id:
            st.session_state.text1 = read_uploaded_file(uploaded_file1)
            st.session_state.current_file1_file_id = uploaded_file1.file_id
    elif st.session_state.current_file1_file_id is not None and uploaded_file1 is None:
        st.session_state.text1 = ""
        st.session_state.current_file1_file_id = None
    st.session_state.text1 = st.text_area("Paste Text/Code 1", value=st.session_state.text1, height=300, key="text_area1", label_visibility="collapsed")

with col2:
    st.subheader("Modified (Right)")
    uploaded_file2 = st.file_uploader("Upload File 2", type=ALLOWED_FILE_TYPES, key="uploader2", help="Upload the second file")
    if uploaded_file2 is not None:
        if st.session_state.current_file2_file_id != uploaded_file2.file_id:
            st.session_state.text2 = read_uploaded_file(uploaded_file2)
            st.session_state.current_file2_file_id = uploaded_file2.file_id
    elif st.session_state.current_file2_file_id is not None and uploaded_file2 is None:
        st.session_state.text2 = ""
        st.session_state.current_file2_file_id = None
    st.session_state.text2 = st.text_area("Paste Text/Code 2", value=st.session_state.text2, height=300, key="text_area2", label_visibility="collapsed")

# --- Options & Actions ---
st.subheader("Options & Actions")
# --- MODIFICATION: Added column for new option ---
opts_col1, opts_col2, opts_col3, opts_col4, clear_col = st.columns([2, 2, 2, 3, 1]) # Adjust ratios

with opts_col1:
    ignore_whitespace = st.checkbox("Ignore leading/trailing whitespace", key="opt_ignore_ws", help="Compares lines after removing leading/trailing spaces and tabs.")
with opts_col2:
    case_insensitive = st.checkbox("Case Insensitive", key="opt_case", help="Performs comparison ignoring text case.")
with opts_col3:
    show_whitespace = st.checkbox("Show Whitespace", key="opt_show_ws", help="Makes spaces (·) and tabs (→) visible.")
# --- MODIFICATION: Add new checkbox for intra-line diff ---
with opts_col4:
    enable_intra_line = st.checkbox("Enable detailed intra-line diff", key="opt_intra_line", value=True, help="Highlights character changes within modified lines. Disable for faster comparison on large files.")
# --- End Modification ---
with clear_col:
    if st.button("Clear All", key="clear_btn", help="Clears inputs and resets file uploads."):
        st.session_state.text1 = ""
        st.session_state.text2 = ""
        st.session_state.current_file1_file_id = None
        st.session_state.current_file2_file_id = None
        st.rerun()

# --- Comparison Logic and Display ---
st.subheader("Comparison Result")
current_text1 = st.session_state.text1
current_text2 = st.session_state.text2

if current_text1 or current_text2:
    with st.spinner("Comparing differences..."):
        try:
            # --- MODIFICATION: Pass new option to get_diff_lines ---
            lines_left, lines_right = get_diff_lines(
                current_text1, current_text2,
                ignore_whitespace=ignore_whitespace,
                case_insensitive=case_insensitive,
                show_whitespace=show_whitespace,
                enable_intra_line_diff=enable_intra_line # Pass the checkbox value
            )
            # --- End Modification ---

            # Construct and render the entire diff view HTML at once
            left_pane_html = render_diff_pane(lines_left)
            right_pane_html = render_diff_pane(lines_right)
            diff_view_html = f"""
            <div class="diff-container">
                <div class="diff-pane">
                    {left_pane_html}
                </div>
                <div class="diff-pane">
                    {right_pane_html}
                </div>
            </div>
            """
            st.markdown(diff_view_html, unsafe_allow_html=True)

            # Updated Legend
            ws_legend = f'(<span class="ws-space">·</span>&nbsp;=&nbsp;Space,&nbsp;<span class="ws-tab">→</span>&nbsp;&nbsp;&nbsp;=&nbsp;Tab)' if show_whitespace else ''
            st.markdown(f"""
            <hr style='margin-top: 1.5rem; margin-bottom: 1rem;'>
            **Legend:**&nbsp;
            <span style='background-color:#d1f7d8; padding: 2px 5px; border-radius: 3px; margin-right: 5px; display: inline-block;'>Added line</span>
            <span style='background-color:#fde2e4; padding: 2px 5px; border-radius: 3px; margin-right: 5px; display: inline-block;'>Deleted line</span>
            <span style='background-color:#fff3cd; padding: 2px 5px; border-radius: 3px; margin-right: 5px; display: inline-block;'>Modified line</span>
            (<span class='char-insert' style='padding: 0 2px;'>char insert</span> / <span class='char-delete' style='padding: 0 2px;'>char delete</span> within line)
            <span style='background-color:#f8f9fa; color: #f8f9fa; padding: 2px 5px; border-radius: 3px; user-select: none; display: inline-block;'>Placeholder</span>
            &nbsp;{ws_legend}
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during comparison: {e}")
            st.exception(e) # Show detailed traceback for debugging
else:
    st.info("Upload files or paste content into the text areas above to start the comparison.")

