# main_app.py
"""
Streamlit application main script.
Orchestrates UI, state management, and calls to processing modules.
"""
import streamlit as st
import os

# Import modules
# Make sure these files exist in the same directory
try:
    import config
    import utils
    import file_processing
    import llm_integration
    import excel_export
    import ui_components
except ModuleNotFoundError as e:
     st.error(f"ERROR: Failed to import a required module: {e}. Ensure all .py files (config, utils, etc.) are present.")
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


# --- Initialize Session State ---
def init_session_state():
    # General App State
    if 'uploaded_file_state' not in st.session_state: st.session_state.uploaded_file_state = None
    if 'extracted_text' not in st.session_state: st.session_state.extracted_text = ""
    if 'identified_applications' not in st.session_state: st.session_state.identified_applications = []
    if 'selected_applications' not in st.session_state: st.session_state.selected_applications = []
    if 'generated_test_cases' not in st.session_state: st.session_state.generated_test_cases = {}
    if 'current_file_identifier' not in st.session_state: st.session_state.current_file_identifier = None
    if 'context_file_selections' not in st.session_state: st.session_state.context_file_selections = {}
    if 'available_context_files' not in st.session_state: st.session_state.available_context_files = [config.NO_CONTEXT_OPTION]

    # LLM Config State
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = list(config.LLM_PROVIDER_CONFIG.keys())[0]
    if 'api_credentials' not in st.session_state: st.session_state.api_credentials = {}
    if 'model_name' not in st.session_state:
         default_models = config.LLM_PROVIDER_CONFIG.get(st.session_state.llm_provider, {}).get("models", [])
         st.session_state.model_name = default_models[0] if default_models else None
    if 'openai_fallback_api_key' not in st.session_state: st.session_state.openai_fallback_api_key = ""

    # Modification State
    if 'modification_app_name' not in st.session_state: st.session_state.modification_app_name = None
    if 'modification_tc_id' not in st.session_state: st.session_state.modification_tc_id = None
    if 'proposed_modification_data' not in st.session_state: st.session_state.proposed_modification_data = None
    if 'original_tc_data_for_diff' not in st.session_state: st.session_state.original_tc_data_for_diff = None
    if 'refactor_request' not in st.session_state: st.session_state.refactor_request = None

    # *** Initialize Log Messages List ***
    if 'log_messages' not in st.session_state: st.session_state.log_messages = []

init_session_state()

# --- Sidebar ---
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload Requirements (.docx)",
        type=config.ACCEPTED_FILE_TYPES,
        key="sidebar_file_uploader_widget",
        help=f"Upload a requirements file in {', '.join(f'.{ext}' for ext in config.ACCEPTED_FILE_TYPES)} format."
    )
    if uploaded_file is not None:
        st.session_state.uploaded_file_state = uploaded_file
    current_file = st.session_state.uploaded_file_state

    st.divider()
    # Render LLM Configuration UI
    ui_components.render_llm_config_sidebar()
    st.divider()
    # Render Context Folder Info
    ui_components.render_context_options_sidebar()
    st.divider()
    st.info("ℹ️ AI results may require review. Always validate generated test cases.")

# --- Main Content Area ---
st.title(config.APP_TITLE)
st.markdown("Upload requirements, configure LLM, identify applications, generate test cases, refactor, and view logs.")

# --- File Processing Logic ---
if current_file:
    current_file_id = (current_file.name, current_file.size)
    if st.session_state.current_file_identifier != current_file_id:
        utils.log_message(f"New file detected: '{current_file.name}'. Resetting state.", "INFO")
        st.info(f"New file detected: '{current_file.name}'. Resetting application state...")
        # Reset state variables
        st.session_state.extracted_text = ""
        st.session_state.identified_applications = []
        st.session_state.selected_applications = []
        st.session_state.generated_test_cases = {}
        st.session_state.context_file_selections = {}
        st.session_state.available_context_files = [config.NO_CONTEXT_OPTION]
        st.session_state.current_file_identifier = current_file_id
        st.session_state.modification_app_name = None
        st.session_state.modification_tc_id = None
        st.session_state.proposed_modification_data = None
        st.session_state.original_tc_data_for_diff = None
        st.session_state.refactor_request = None
        st.session_state.log_messages = [] # Clear logs for new file
        st.rerun()

    # --- Text Extraction ---
    if not st.session_state.extracted_text:
        utils.log_message(f"Attempting text extraction from '{current_file.name}'.", "INFO")
        with st.spinner(f"Extracting text from '{current_file.name}'..."):
            extracted = file_processing.extract_text_from_docx(current_file)
            if extracted is not None: # Check for None explicitly
                st.session_state.extracted_text = extracted
                utils.log_message(f"Text extracted successfully ({len(extracted)} chars).", "INFO")
                st.session_state.available_context_files = utils.get_available_context_files()
                st.success("Text extracted successfully.")
                st.rerun()
            else:
                utils.log_message("Text extraction failed.", "ERROR")
                # Error message handled by extract_text_from_docx
                st.stop()

    # --- Main Workflow Tabs (only if text is available) ---
    if st.session_state.extracted_text:
        # *** Added tab_log ***
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
                st.session_state.context_file_selections = {}
                st.session_state.modification_app_name = None; st.session_state.modification_tc_id = None
                st.session_state.proposed_modification_data = None; st.session_state.original_tc_data_for_diff = None
                st.session_state.refactor_request = None

                creds_ok, creds_msg = llm_integration.check_credentials(
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
                    llm, _ = llm_integration.get_llm_and_embeddings(
                        st.session_state.llm_provider, st.session_state.model_name,
                        st.session_state.api_credentials, st.session_state.openai_fallback_api_key
                    )
                    if llm:
                        identified = llm_integration.identify_applications(st.session_state.extracted_text, llm)
                        st.session_state.identified_applications = identified
                        if identified:
                            st.success(f"Identified {len(identified)} potential applications.")
                            st.session_state.available_context_files = utils.get_available_context_files()
                            st.session_state.context_file_selections = { app: config.NO_CONTEXT_OPTION for app in identified }
                            st.rerun()
                        else:
                             st.warning("No applications identified by the LLM.")
                             utils.log_message("Identification finished, but no applications found.", "INFO")
                    else:
                         utils.log_message("Identify failed: LLM initialization failed.", "ERROR")
                         st.error("LLM initialization failed. Cannot identify applications. Check configuration and credentials in sidebar and logs.")

            # --- Step 3: Select Apps & Context ---
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

                    creds_ok, creds_msg = llm_integration.check_credentials(
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
                        llm, embeddings = llm_integration.get_llm_and_embeddings(
                            st.session_state.llm_provider, st.session_state.model_name,
                            st.session_state.api_credentials, st.session_state.openai_fallback_api_key
                        )
                        if llm and embeddings:
                            utils.log_message("LLM and Embeddings ready, calling generate_test_cases...", "INFO")
                            results = llm_integration.generate_test_cases(
                                st.session_state.extracted_text, st.session_state.selected_applications,
                                st.session_state.context_file_selections, llm, embeddings
                            )
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
            if st.session_state.get('refactor_request'): # Use .get for safety
                 req = st.session_state.refactor_request
                 st.session_state.refactor_request = None # Consume the request
                 utils.log_message(f"Processing refactor request for TC '{req['tc_id']}' in app '{req['app_name']}'.", "INFO")

                 creds_ok, creds_msg = llm_integration.check_credentials(
                     st.session_state.llm_provider, st.session_state.api_credentials,
                     st.session_state.openai_fallback_api_key, require_fallback_for_rag=False
                 )
                 if not creds_ok:
                     utils.log_message(f"Refactor failed: Credentials check failed - {creds_msg}", "ERROR")
                     st.error(f"Cannot refactor: {creds_msg}")
                 elif not st.session_state.get("model_name"):
                     utils.log_message(f"Refactor failed: No model selected for {st.session_state.llm_provider}.", "ERROR")
                     st.error(f"Cannot refactor: No model selected for {st.session_state.llm_provider}.")
                 else:
                     llm, _ = llm_integration.get_llm_and_embeddings(
                         st.session_state.llm_provider, st.session_state.model_name,
                         st.session_state.api_credentials, st.session_state.openai_fallback_api_key
                     )
                     if llm:
                          with st.spinner(f"Refactoring Test Case '{req['tc_id']}'..."):
                                refactored_data = llm_integration.refactor_single_test_case(
                                    req['app_name'], req['tc_id'], req['instructions'],
                                    req['original_data'], llm
                                )
                          if refactored_data:
                                st.session_state.modification_app_name = req['app_name']
                                st.session_state.modification_tc_id = req['tc_id']
                                st.session_state.proposed_modification_data = refactored_data
                                st.session_state.original_tc_data_for_diff = req['original_data']
                                utils.log_message(f"Refactoring successful for TC '{req['tc_id']}'. Awaiting confirmation.", "INFO")
                                st.success("Refactoring complete. Review the proposed changes below.")
                                st.rerun()
                          else:
                               utils.log_message(f"Refactoring failed for TC '{req['tc_id']}'. LLM did not return valid data.", "ERROR")
                               st.error("Refactoring failed. LLM did not return valid data. Check logs.")
                     else:
                          utils.log_message("Refactor failed: LLM could not be initialized.", "ERROR")
                          st.error("Refactoring failed: LLM could not be initialized. Check logs.")

            # --- Display Current Results (for context) ---
            if st.session_state.generated_test_cases:
                st.subheader("Current Test Cases (including applied modifications)")
                ui_components.display_results(st.session_state.generated_test_cases)
                st.markdown("---")
                # --- Render Modification UI ---
                if st.session_state.proposed_modification_data:
                    ui_components.render_modification_confirmation_ui()
                else:
                    ui_components.render_modification_request_ui()
            else:
                st.info("⬅️ Generate test cases first before attempting to refactor.")

        # === Log Tab ===
        with tab_log:
            st.header("📜 Session Log")
            st.caption("Debug messages and errors for the current session are logged here. Newest messages are at the top.")

            if st.button("Clear Log", key="clear_log_btn"):
                st.session_state.log_messages = []
                utils.log_message("Log cleared by user.", "INFO") # Log the clear action itself
                st.rerun()

            log_area_height = 400 # Adjust height as needed
            if 'log_messages' in st.session_state and st.session_state.log_messages:
                # Display logs in reverse order (newest first) in a scrollable container
                log_container = st.container(height=log_area_height)
                with log_container:
                    # Use st.text for simple, preformatted display
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
    st.caption(f"Optional: Create a folder named `{config.APP_CONTEXT_FOLDER_NAME}` in `{os.getcwd()}` to add `.yaml`/`.yml` context files.")

