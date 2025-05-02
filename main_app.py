"""
Streamlit application main script.
Orchestrates UI, state management, and calls to processing modules.
"""
import streamlit as st
import os
# import json # No longer needed for this clipboard method

# *** Import streamlit-clipboard ***
try:
    # Use the specific component name as imported in your original code
    from clipboard_component import copy_component
except ImportError:
    st.error("ERROR: `streamlit-clipboard` library or `clipboard_component` not found. Please ensure it's installed and accessible.")
    st.stop()


# Import modules
# Make sure these files exist in the same directory or are in PYTHONPATH
try:
    import config
    # Assuming helper modules are in a 'helper' subfolder
    import helper.utils as utils
    import helper.file_processing as file_processing
    import helper.excel_export as excel_export
    # Assuming llm_integration_core and ui_components are at the same level or accessible
    import llm_integration_core
    import ui_components
except ModuleNotFoundError as e:
     st.error(f"ERROR: Failed to import a required module: {e}. Ensure all .py files (config, utils, etc.) and folders (helper, llm_providers) are present relative to main_app.py.")
     st.stop()
except Exception as e:
     st.error(f"ERROR: An unexpected error occurred during module imports: {e}")
     st.stop()

# --- Page Configuration (Must be the first Streamlit command) ---
try:
    st.set_page_config(page_title=config.APP_TITLE, layout=config.PAGE_LAYOUT)
except st.errors.StreamlitAPIException as e:
    if "can only be called once per app" not in str(e).lower(): raise e
except Exception as e: # Catch other potential errors during set_page_config
     st.error(f"Error setting page config: {e}")

# --- Custom CSS Injection ---
# (CSS remains the same - omitted for brevity but should be included in your actual file)
st.markdown("""
<style>
    /* Your CSS rules here */
    html, body {
        font-size: 0.85rem !important;
    }
    button[data-testid="stTab"] {
        font-size: 4rem !important;
        padding: 0rem 0rem !important;
    }
    button[data-testid="stTab"] p {
        font-size: 1rem !important;
        margin: 0.3 !important;
        font-weight: bold !important;
    }
    code {
        color: #00008B !important;
        font-size: 1.05em !important;
    }
</style>
""", unsafe_allow_html=True)
# --- End Custom CSS ---

# --- Initialize Session State ---
def init_session_state():
    # General App State
    if 'uploaded_file_state' not in st.session_state: st.session_state.uploaded_file_state = None
    if 'extracted_text' not in st.session_state: st.session_state.extracted_text = ""
    if 'identified_applications' not in st.session_state: st.session_state.identified_applications = []
    if 'selected_applications' not in st.session_state: st.session_state.selected_applications = []
    if 'generated_test_cases' not in st.session_state: st.session_state.generated_test_cases = {}
    if 'current_file_identifier' not in st.session_state: st.session_state.current_file_identifier = None

    # *** MODIFIED/NEW State for Context Handling ***
    # Remove or comment out old state if desired
    # if 'context_file_selections' in st.session_state: del st.session_state['context_file_selections']
    # if 'available_context_files' in st.session_state: del st.session_state['available_context_files']
    # Add new state for uploaded context files (maps app_name to list of UploadedFile objects)
    if 'uploaded_context_files' not in st.session_state: st.session_state.uploaded_context_files = {}
    # Keep available_context_files if the sidebar still uses it for info
    if 'available_context_files' not in st.session_state: st.session_state.available_context_files = [config.NO_CONTEXT_OPTION]
    # *** END MODIFICATION ***

    # LLM Config State
    if 'llm_provider' not in st.session_state:
        # Ensure LLM_PROVIDER_CONFIG is not empty before accessing keys
        provider_keys = list(config.LLM_PROVIDER_CONFIG.keys())
        st.session_state.llm_provider = provider_keys[0] if provider_keys else None
    if 'api_credentials' not in st.session_state: st.session_state.api_credentials = {}
    if 'model_name' not in st.session_state:
        # Safely get default models, handle missing provider or models key
        default_models = config.LLM_PROVIDER_CONFIG.get(st.session_state.get('llm_provider', ''), {}).get("models", [])
        st.session_state.model_name = default_models[0] if default_models else None
    if 'openai_fallback_api_key' not in st.session_state: st.session_state.openai_fallback_api_key = ""

    # Modification State (Single Case)
    if 'modification_app_name' not in st.session_state: st.session_state.modification_app_name = None
    if 'modification_tc_id' not in st.session_state: st.session_state.modification_tc_id = None
    if 'proposed_modification_data' not in st.session_state: st.session_state.proposed_modification_data = None
    if 'original_tc_data_for_diff' not in st.session_state: st.session_state.original_tc_data_for_diff = None
    if 'refactor_request' not in st.session_state: st.session_state.refactor_request = None

    # Modification State (Bulk Case) - NEW
    if 'refactor_all_request' not in st.session_state: st.session_state.refactor_all_request = None
    if 'refactored_test_cases' not in st.session_state: st.session_state.refactored_test_cases = None

    # Initialize Log Messages List
    if 'log_messages' not in st.session_state: st.session_state.log_messages = []

init_session_state()

# --- Sidebar ---
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload Requirements (.docx)",
        type=config.ACCEPTED_FILE_TYPES,
        key="sidebar_file_uploader_widget", # Keep key consistent
        help=f"Upload a requirements file in {', '.join(f'.{ext}' for ext in config.ACCEPTED_FILE_TYPES)} format."
    )
    # Update state only if a new file is uploaded
    if uploaded_file is not None and st.session_state.uploaded_file_state != uploaded_file:
         st.session_state.uploaded_file_state = uploaded_file
         # No automatic rerun here, let the main logic handle new file detection

    current_file = st.session_state.uploaded_file_state # Use the potentially updated state

    st.divider()
    # Render LLM Configuration UI
    ui_components.render_llm_config_sidebar()
    st.divider()
    # Render Context Folder Info (might be less relevant now but kept for info)
    ui_components.render_context_options_sidebar()
    st.divider()
    st.info("ℹ️ AI results may require review. Always validate generated test cases.")


# --- Main Content Area ---
st.title(config.APP_TITLE)
st.markdown("Upload requirements, configure LLM, identify applications, generate test cases, refactor, and view logs.")

# --- File Processing Logic ---
if current_file:
    # Use file name and size as a unique identifier for the uploaded requirements doc
    current_file_id = (current_file.name, current_file.size)

    # Check if the file has changed since the last run
    if st.session_state.current_file_identifier != current_file_id:
        utils.log_message(f"New file detected: '{current_file.name}'. Resetting state.", "INFO")
        st.info(f"New file detected: '{current_file.name}'. Resetting application state...")
        # Reset state variables associated with the previous file
        st.session_state.extracted_text = ""
        st.session_state.identified_applications = []
        st.session_state.selected_applications = []
        st.session_state.generated_test_cases = {}

        # *** MODIFIED: Reset new context state ***
        st.session_state.uploaded_context_files = {} # Clear uploaded context files
        # Remove old state reset if desired
        # st.session_state.context_file_selections = {}
        # st.session_state.available_context_files = [config.NO_CONTEXT_OPTION] # Likely not needed here anymore
        # *** END MODIFICATION ***

        st.session_state.current_file_identifier = current_file_id # Store the new identifier
        # Reset single modification state
        st.session_state.modification_app_name = None
        st.session_state.modification_tc_id = None
        st.session_state.proposed_modification_data = None
        st.session_state.original_tc_data_for_diff = None
        st.session_state.refactor_request = None
        # Reset bulk modification state
        st.session_state.refactor_all_request = None
        st.session_state.refactored_test_cases = None

        st.session_state.log_messages = [] # Clear logs for new file
        st.rerun() # Rerun immediately to process the new file

    # --- Text Extraction (only if not already extracted for current file) ---
    if not st.session_state.extracted_text:
        utils.log_message(f"Attempting text extraction from '{current_file.name}'.", "INFO")
        with st.spinner(f"Extracting text from '{current_file.name}'..."):
            # Use the generic dispatcher if available, otherwise call specific extractor
            if hasattr(file_processing, 'extract_text_from_file'):
                 extracted = file_processing.extract_text_from_file(current_file)
            elif current_file.name.lower().endswith(".docx"):
                 extracted = file_processing.extract_text_from_docx(current_file)
            else:
                 extracted = None # Handle other types if needed or show error
                 utils.log_message(f"Unsupported main file type for direct extraction: {current_file.name}", "ERROR")
                 st.error(f"Unsupported file type for requirements document: {current_file.name}. Please upload a .docx file.")


            if extracted is not None: # Check for None explicitly
                st.session_state.extracted_text = extracted
                utils.log_message(f"Text extracted successfully ({len(extracted)} chars).", "INFO")
                # Update available context files list (for sidebar info) if needed
                # This might be removed if sidebar info is removed
                try:
                    st.session_state.available_context_files = utils.get_available_context_files()
                except Exception as e:
                     utils.log_message(f"Could not update available context files list: {e}", "WARNING")
                st.success("Text extracted successfully.")
                st.rerun() # Rerun to display tabs now that text is available
            else:
                # Error message should have been shown by the extraction function or the check above
                utils.log_message("Text extraction failed.", "ERROR")
                st.stop() # Stop if main text extraction fails


    # --- Main Workflow Tabs (only if text is available) ---
    if st.session_state.extracted_text:
        # Added tab_log
        tab1, tab2, tab_log = st.tabs([
            "Generate Test Cases",
            "Refactor Test Cases",
            "📜 Session Logs"
        ])

        # === Generate Tab ===
        with tab1:
            st.header("Generation Workflow")

            # --- Step 2: Identify Applications ---
            st.subheader("2. Identify Applications")
            if st.button("Identify Applications", key="identify_btn"):
                utils.log_message("'Identify Applications' button clicked.", "INFO")
                # Reset downstream state
                st.session_state.identified_applications = []
                st.session_state.selected_applications = []
                st.session_state.generated_test_cases = {}
                # *** MODIFIED: Reset new context state ***
                st.session_state.uploaded_context_files = {}
                # Remove old state reset if desired
                # st.session_state.context_file_selections = {}
                # *** END MODIFICATION ***
                # Reset modification states
                st.session_state.modification_app_name = None; st.session_state.modification_tc_id = None
                st.session_state.proposed_modification_data = None; st.session_state.original_tc_data_for_diff = None
                st.session_state.refactor_request = None
                st.session_state.refactor_all_request = None
                st.session_state.refactored_test_cases = None

                creds_ok, creds_msg = llm_integration_core.check_credentials(
                    st.session_state.llm_provider, st.session_state.api_credentials,
                    st.session_state.openai_fallback_api_key, require_fallback_for_rag=False
                )
                if not creds_ok:
                    utils.log_message(f"Identify failed: Credentials check failed - {creds_msg}", "WARNING")
                    st.warning(f"Cannot identify: {creds_msg}")
                elif not st.session_state.get("model_name"): # Use .get for safety
                    utils.log_message(f"Identify failed: No model selected for {st.session_state.llm_provider}.", "WARNING")
                    st.warning(f"Cannot identify: No model selected for {st.session_state.llm_provider}.")
                else:
                    llm, _ = llm_integration_core.get_llm_and_embeddings(
                        st.session_state.llm_provider, st.session_state.model_name,
                        st.session_state.api_credentials, st.session_state.openai_fallback_api_key
                    )
                    if llm:
                        identified = llm_integration_core.identify_applications(
                            st.session_state.extracted_text,
                            llm,
                            st.session_state.llm_provider # Pass the provider name
                        )
                        st.session_state.identified_applications = identified
                        if identified:
                            st.success(f"Identified {len(identified)} potential applications.")
                            # Initialize uploaded context files state for newly identified apps
                            st.session_state.uploaded_context_files = { app: [] for app in identified }
                            # Update available context files list (for sidebar info) if needed
                            try:
                                st.session_state.available_context_files = utils.get_available_context_files()
                            except Exception as e:
                                utils.log_message(f"Could not update available context files list after identification: {e}", "WARNING")
                            st.rerun()
                        else:
                            st.warning("No applications identified by the LLM.")
                            utils.log_message("Identification finished, but no applications found.", "INFO")
                    else:
                        utils.log_message("Identify failed: LLM initialization failed.", "ERROR")
                        st.error("LLM initialization failed. Cannot identify applications. Check configuration and credentials in sidebar and logs.")

            # --- Step 3: Select Apps & Upload Context ---
            # This function now handles the file uploaders
            ui_components.render_app_context_selection(st.session_state.identified_applications)

            # --- Step 4: Generate Button & Logic ---
            if st.session_state.identified_applications and st.session_state.selected_applications:
                st.markdown("---")
                st.subheader("4. Generate Test Cases")
                num_selected = len(st.session_state.selected_applications)
                if st.button(f"Generate Cases for {num_selected} Application(s)", key="generate_btn", type="primary"):
                    utils.log_message(f"'Generate Cases' button clicked for apps: {st.session_state.selected_applications}", "INFO")
                    # Clear previous results and modification state
                    st.session_state.generated_test_cases = {}
                    st.session_state.modification_app_name = None; st.session_state.modification_tc_id = None
                    st.session_state.proposed_modification_data = None; st.session_state.original_tc_data_for_diff = None
                    st.session_state.refactor_request = None
                    st.session_state.refactor_all_request = None
                    st.session_state.refactored_test_cases = None

                    creds_ok, creds_msg = llm_integration_core.check_credentials(
                        st.session_state.llm_provider, st.session_state.api_credentials,
                        st.session_state.openai_fallback_api_key, require_fallback_for_rag=True
                    )
                    if not creds_ok:
                        utils.log_message(f"Generate failed: Credentials check failed - {creds_msg}", "ERROR")
                        st.error(f"Cannot generate: {creds_msg}")
                    elif not st.session_state.get("model_name"):
                        utils.log_message(f"Generate failed: No model selected for {st.session_state.llm_provider}.", "ERROR")
                        st.error(f"Cannot generate: No model selected for {st.session_state.llm_provider}.")
                    else:
                        st.info(f"Initializing generation using {st.session_state.llm_provider} ({st.session_state.model_name})...")
                        llm, embeddings = llm_integration_core.get_llm_and_embeddings(
                            st.session_state.llm_provider, st.session_state.model_name,
                            st.session_state.api_credentials, st.session_state.openai_fallback_api_key
                        )
                        if llm and embeddings:
                            utils.log_message("LLM and Embeddings ready, calling generate_test_cases...", "INFO")

                            # *** MODIFIED CALL: Pass uploaded_context_files ***
                            results = llm_integration_core.generate_test_cases(
                                st.session_state.extracted_text,
                                st.session_state.selected_applications,
                                st.session_state.uploaded_context_files, # Pass the dict of uploaded file lists
                                llm,
                                embeddings,
                                st.session_state.llm_provider
                            )
                            # *** END MODIFICATION ***

                            st.session_state.generated_test_cases = results
                            utils.log_message(f"Generation process complete. Result keys: {list(results.keys())}", "INFO")
                            if results: st.success("Test case generation process complete. Check results below or logs for details.")
                            else: st.warning("Generation process finished, but no results were returned. Check logs.")
                            st.rerun() # Update UI to show results/errors immediately
                        elif not llm:
                            utils.log_message("Generate failed: LLM initialization failed.", "ERROR")
                            st.error("Generation failed: LLM could not be initialized. Check logs.")
                        else: # Embeddings must have failed
                            utils.log_message("Generate failed: Embeddings initialization failed.", "ERROR")
                            st.error("Generation failed: Embeddings could not be initialized (required for RAG). Check logs.")

            # --- Display Generated Results ---
            if st.session_state.generated_test_cases:
                st.markdown("---")
                st.header("Generated Results")
                ui_components.display_results(st.session_state.generated_test_cases)


        # === Refactor Tab ===
        with tab2:
            st.header("Refactor Generated Test Cases")

            # --- Handle Refactoring Request ---
            # Check for single refactor request first
            if st.session_state.get('refactor_request'):
                req = st.session_state.refactor_request
                st.session_state.refactor_request = None # Consume the request
                utils.log_message(f"Processing single refactor request for TC '{req['tc_id']}' in app '{req['app_name']}'.", "INFO")

                # Clear bulk state if starting single refactor
                st.session_state.refactor_all_request = None
                st.session_state.refactored_test_cases = None

                creds_ok, creds_msg = llm_integration_core.check_credentials(
                    st.session_state.llm_provider, st.session_state.api_credentials,
                    st.session_state.openai_fallback_api_key, require_fallback_for_rag=False
                )
                if not creds_ok:
                    utils.log_message(f"Single refactor failed: Credentials check failed - {creds_msg}", "ERROR")
                    st.error(f"Cannot refactor: {creds_msg}")
                elif not st.session_state.get("model_name"):
                    utils.log_message(f"Single refactor failed: No model selected for {st.session_state.llm_provider}.", "ERROR")
                    st.error(f"Cannot refactor: No model selected for {st.session_state.llm_provider}.")
                else:
                    llm, _ = llm_integration_core.get_llm_and_embeddings(
                        st.session_state.llm_provider, st.session_state.model_name,
                        st.session_state.api_credentials, st.session_state.openai_fallback_api_key
                    )
                    if llm:
                        with st.spinner(f"Refactoring Test Case '{req['tc_id']}'..."):
                            refactored_data = llm_integration_core.refactor_single_test_case(
                                req['app_name'], req['tc_id'], req['instructions'],
                                req['original_data'], llm,
                                st.session_state.llm_provider # Pass the provider name
                            )
                        if refactored_data:
                            st.session_state.modification_app_name = req['app_name']
                            st.session_state.modification_tc_id = req['tc_id']
                            st.session_state.proposed_modification_data = refactored_data
                            st.session_state.original_tc_data_for_diff = req['original_data']
                            utils.log_message(f"Single refactoring successful for TC '{req['tc_id']}'. Awaiting confirmation.", "INFO")
                            st.success("Refactoring complete. Review the proposed changes below.")
                        else:
                            utils.log_message(f"Single refactoring failed for TC '{req['tc_id']}'. LLM did not return valid data.", "ERROR")
                            st.error("Refactoring failed. LLM did not return valid data. Check logs.")
                            # Put request back so user can see the error context? Or just clear? Clear for now.
                            st.session_state.modification_app_name = None
                            st.session_state.modification_tc_id = None
                            st.session_state.proposed_modification_data = None
                            st.session_state.original_tc_data_for_diff = None
                    else:
                        utils.log_message("Single refactor failed: LLM could not be initialized.", "ERROR")
                        st.error("Refactoring failed: LLM could not be initialized. Check logs.")
                st.rerun() # Rerun after processing single request

            # Check for bulk refactor request
            elif st.session_state.get('refactor_all_request'):
                req = st.session_state.refactor_all_request
                # Consume the request *after* LLM call, so confirmation UI can use it
                utils.log_message(f"Processing bulk refactor request for app '{req['app_name']}'.", "INFO")

                # Clear single modification state if starting bulk refactor
                st.session_state.modification_app_name = None
                st.session_state.modification_tc_id = None
                st.session_state.proposed_modification_data = None
                st.session_state.original_tc_data_for_diff = None
                st.session_state.refactored_test_cases = None # Clear previous bulk results before starting

                creds_ok, creds_msg = llm_integration_core.check_credentials(
                    st.session_state.llm_provider, st.session_state.api_credentials,
                    st.session_state.openai_fallback_api_key, require_fallback_for_rag=False # RAG not needed
                )
                if not creds_ok:
                    utils.log_message(f"Bulk refactor failed: Credentials check failed - {creds_msg}", "ERROR")
                    st.error(f"Cannot refactor: {creds_msg}")
                    # Keep request in state so confirmation UI shows error context
                elif not st.session_state.get("model_name"):
                    utils.log_message(f"Bulk refactor failed: No model selected for {st.session_state.llm_provider}.", "ERROR")
                    st.error(f"Cannot refactor: No model selected for {st.session_state.llm_provider}.")
                    # Keep request in state
                else:
                    llm, _ = llm_integration_core.get_llm_and_embeddings(
                        st.session_state.llm_provider, st.session_state.model_name,
                        st.session_state.api_credentials, st.session_state.openai_fallback_api_key
                    )
                    if llm:
                        with st.spinner(f"Refactoring all test cases for '{req['app_name']}'..."):
                            # *** Call the NEW bulk refactoring function ***
                            # Ensure the function exists in llm_integration_core
                            if hasattr(llm_integration_core, 'refactor_all_test_cases'):
                                refactored_list = llm_integration_core.refactor_all_test_cases(
                                    req['app_name'], req['instructions'],
                                    req['original_data'], # Pass the list of original cases
                                    llm,
                                    st.session_state.llm_provider # Pass the provider name
                                )
                            else:
                                utils.log_message("Bulk refactor failed: `refactor_all_test_cases` function not found in llm_integration_core.py.", "ERROR")
                                st.error("Internal Error: Bulk refactoring function is missing.")
                                refactored_list = None # Indicate failure

                        if refactored_list is not None: # Check if it returned something (could be empty list on success)
                            st.session_state.refactored_test_cases = refactored_list # Store the list of results
                            utils.log_message(f"Bulk refactoring successful for app '{req['app_name']}'. Found {len(refactored_list)} cases. Awaiting confirmation.", "INFO")
                            st.success("Refactoring complete. Review the proposed changes below.")
                            # Keep request in state for confirmation UI
                        else:
                            utils.log_message(f"Bulk refactoring failed for app '{req['app_name']}'. LLM did not return valid data or function missing.", "ERROR")
                            st.error("Refactoring failed. LLM did not return a valid list of test cases or function missing. Check logs.")
                            # Keep request in state for confirmation UI
                    else:
                        utils.log_message("Bulk refactor failed: LLM could not be initialized.", "ERROR")
                        st.error("Refactoring failed: LLM could not be initialized. Check logs.")
                        # Keep request in state for confirmation UI

                # REMOVED: st.session_state.refactor_all_request = req (State persists naturally)
                # REMOVED: st.rerun() (Rerun will happen automatically, allowing confirmation UI to render)

            # --- Display Current Results (for context) ---
            if st.session_state.generated_test_cases:
                st.subheader("Current Test Cases (including applied modifications)")
                ui_components.display_results(st.session_state.generated_test_cases)
                st.markdown("---")

                # --- Render Modification UI ---
                # Render confirmation if either single proposed data or bulk request (with or without results) exists
                if st.session_state.get('proposed_modification_data') or st.session_state.get('refactor_all_request'):
                    ui_components.render_modification_confirmation_ui()
                else:
                    # Otherwise, render the request UI (which is now the bulk request UI)
                    ui_components.render_modification_request_ui()
            else:
                st.info("⬅️ Generate test cases first before attempting to refactor.")


        # === Log Tab ===
        with tab_log:
            st.header("📜 Session Log")
            st.caption("Debug messages and errors for the current session are logged here. Newest messages are at the top.")

            # --- Action Buttons ---
            col1, col2, col_spacer = st.columns([1, 1, 4]) # Use columns for layout

            with col1:
                if st.button("Clear Log", key="clear_log_btn"):
                    st.session_state.log_messages = []
                    utils.log_message("Log cleared by user.", "INFO") # Log the action itself
                    st.rerun()

            with col2:
                # Prepare log string for copying
                log_string_to_copy = ""
                if 'log_messages' in st.session_state and st.session_state.log_messages:
                    # Join messages with newline for clipboard
                    log_string_to_copy = "\n".join(st.session_state.log_messages)

                copy_button_disabled = not bool(log_string_to_copy)

                # *** Use streamlit_clipboard.copy ***
                copy_component("Copy Log", content=log_string_to_copy, disabled=copy_button_disabled)

            # --- Display Logs ---
            log_area_height = 400 # Adjust height as needed
            if 'log_messages' in st.session_state and st.session_state.log_messages:
                # Display logs in reverse order (newest first) in a scrollable container
                log_container = st.container(height=log_area_height, border=True)
                with log_container:
                    # Use st.text for simple, preformatted display that respects newlines
                    st.text("\n".join(reversed(st.session_state.log_messages)))
            else:
                st.info("Log is empty.")


        # --- Export Button (Outside Tabs) ---
        if st.session_state.generated_test_cases:
            st.markdown("---")
            can_export = any(isinstance(c, list) and c for c in st.session_state.generated_test_cases.values())
            if can_export:
                st.subheader("⬇️ Export Results")
                excel_bytes = None
                try:
                    excel_bytes = excel_export.export_to_excel(st.session_state.generated_test_cases)
                except Exception as e:
                    utils.log_message(f"Excel preparation error: {e}", "ERROR")
                    st.error(f"An unexpected error occurred during Excel preparation: {e}")

                if excel_bytes:
                    st.download_button(
                        label="Download All Test Cases (.xlsx)", data=excel_bytes,
                        file_name=config.EXCEL_EXPORT_FILENAME,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel_btn",
                        help="Downloads all currently displayed test cases to an Excel file."
                    )
                else:
                    st.warning("Excel file generation failed. Cannot download.")


# --- Initial State Message ---
else:
    st.info("⬅️ Upload a `.docx` document using the file uploader in the sidebar to begin.")
    # Updated caption to reflect context upload change
    st.caption("Optional context files (.txt, .md, .docx, .xlsx, .json, .yaml) can be uploaded per application in Step 3 after identifying applications.")

    st.caption("Note: Context files are not required for generation but can improve results.")
