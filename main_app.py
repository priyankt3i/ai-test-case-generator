"""
Streamlit application main script.
Orchestrates UI, state management, and calls to processing modules.
"""
import streamlit as st
import os
import re # Import regex module
# import json # No longer needed for this clipboard method
from PIL import Image

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
    if 'generation_cost' not in st.session_state: st.session_state.generation_cost = 0.0
    if 'token_usage' not in st.session_state: st.session_state.token_usage = {"input": 0, "output": 0}

    # *** MODIFIED/NEW State for Context Handling ***
    # Remove or comment out old state
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

    # --- AI Review Test Cases State ---
    if 'ai_review_selected_app' not in st.session_state: st.session_state.ai_review_selected_app = None
    if 'ai_review_results_raw' not in st.session_state: st.session_state.ai_review_results_raw = None
    if 'ai_review_suggestions_processed' not in st.session_state: st.session_state.ai_review_suggestions_processed = None # Will store structured suggestions
    if 'ai_review_user_decisions' not in st.session_state: st.session_state.ai_review_user_decisions = {} # Maps suggestion_id to 'accept'/'reject'
    if 'ai_review_inprogress_flag' not in st.session_state: st.session_state.ai_review_inprogress_flag = False

init_session_state()

# --- Sidebar ---
with st.sidebar:
    try:
        logo = Image.open("public/logo.png") # Replace with the actual path to your logo
        st.sidebar.image(logo, width=700) # Adjust width as needed
    except FileNotFoundError:
        st.sidebar.error("Logo image not found. Please check the path.")

    st.header("📄 1.A Upload Document")
    uploaded_files = st.file_uploader(
        "Upload Requirements (.docx, .pdf)",
        type=config.ACCEPTED_FILE_TYPES,
        key="sidebar_file_uploader_widget",
        help=f"Upload one or more requirements files in {', '.join(f'.{ext}' for ext in config.ACCEPTED_FILE_TYPES)} format.",
        accept_multiple_files=True
    )

    if uploaded_files:
        st.session_state.uploaded_file_state = uploaded_files
    
    current_files = st.session_state.get('uploaded_file_state', [])

    st.divider()
    # Render LLM Configuration UI
    ui_components.render_llm_config_sidebar()
    st.divider()
    # Render Context Folder Info (might be less relevant now but kept for info)
    #ui_components.render_context_options_sidebar()
    #st.divider()
    st.info("ℹ️ AI results may require review. Always validate generated test cases.")


# --- Main Content Area ---
st.title(config.APP_TITLE)
st.header("Convert Business Requirements to Test Cases")
st.markdown("Upload requirements, configure LLM, identify applications, generate test cases, refactor, and view logs.")

# --- File Processing Logic ---
if current_files:
    # Create a unique identifier for the current set of files
    current_files_id = tuple((file.name, file.size) for file in current_files)

    # Check if the set of files has changed since the last run
    if st.session_state.current_file_identifier != current_files_id:
        utils.log_message(f"New file set detected. Resetting state.", "INFO")
        st.info(f"New file set detected. Resetting application state...")
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

        st.session_state.current_file_identifier = current_files_id # Store the new identifier
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
        all_extracted_text = []
        with st.spinner(f"Extracting text from {len(current_files)} file(s)..."):
            for file in current_files:
                utils.log_message(f"Attempting text extraction from '{file.name}'.", "INFO")
                extracted = file_processing.extract_text_from_file(file)
                if extracted:
                    all_extracted_text.append(extracted)
                else:
                    utils.log_message(f"Text extraction failed for '{file.name}'.", "ERROR")
                    st.error(f"Failed to extract text from '{file.name}'.")
        
        if all_extracted_text:
            extracted = "\n\n---\n\n".join(all_extracted_text)


            if extracted is not None: # Check for None explicitly
                st.session_state.extracted_text = extracted
                utils.log_message(f"Text extracted successfully ({len(extracted)} chars).", "INFO")
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
        tab1, tab2, tab3, tab_log = st.tabs([
            "Generate Test Cases",
            "AI Review Test Cases",
            "Manual Refactor Test Cases",
            "📜 Session Logs"
        ])

        # === Generate Tab ===
        with tab1:
            st.header("Generation Workflow")

            # --- Step 2: Identify Applications ---
            st.subheader("2. Identify Applications")
            if st.button("Identify Applications", key="identify_btn"):
                utils.log_message("'Identify Applications' button clicked.", "INFO")

                st.session_state.identified_applications = []
                st.session_state.selected_applications = []
                st.session_state.generated_test_cases = {}
                st.session_state.uploaded_context_files = {}
                # Remove old state reset if desired
                # st.session_state.context_file_selections = {}
  
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
                elif not st.session_state.get("model_name"): 
                    utils.log_message(f"Identify failed: No model selected for {st.session_state.llm_provider}.", "WARNING")
                    st.warning(f"Cannot identify: No model selected for {st.session_state.llm_provider}.")
                else:
                    llm, _ = llm_integration_core.get_llm_and_embeddings(
                        st.session_state.llm_provider, st.session_state.model_name,
                        st.session_state.api_credentials, st.session_state.openai_fallback_api_key
                    )
                    if llm:
                        identified, token_usage = llm_integration_core.identify_applications(
                            st.session_state.extracted_text,
                            llm,
                            st.session_state.llm_provider
                        )
                        st.session_state.identified_applications = identified
                        if identified:
                            st.success(f"Identified {len(identified)} potential applications.")
                            st.session_state.uploaded_context_files = { app: [] for app in identified }
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
            # This function handles the file uploaders
            ui_components.render_app_context_selection(st.session_state.identified_applications)

            # --- Step 4: Generate Button & Logic ---
            if st.session_state.identified_applications and st.session_state.selected_applications:
                st.markdown("---")
                st.subheader("4. Generate Test Cases")
                num_selected = len(st.session_state.selected_applications)
                if st.button(f"Generate Cases for {num_selected} Application(s)", key="generate_btn", type="primary"):
                    utils.log_message(f"'Generate Cases' button clicked for apps: {st.session_state.selected_applications}", "INFO")

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

                            results, token_usage = llm_integration_core.generate_test_cases(
                                st.session_state.extracted_text,
                                st.session_state.selected_applications,
                                st.session_state.uploaded_context_files,
                                llm,
                                embeddings,
                                st.session_state.llm_provider,
                                st.session_state.model_name
                            )

                            st.session_state.generated_test_cases = results
                            st.session_state.token_usage = token_usage
                            st.session_state.generation_cost = token_usage.get('cost', 0.0)
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

        # === AI Review Test Cases Tab ===
        with tab2:
            st.header("🤖 AI Review Test Cases")
            st.caption("Use AI to review generated test cases for coverage, gaps, and suggest improvements.")

            if not st.session_state.get('generated_test_cases'):
                st.info("⬅️ Please generate test cases in the 'Generate Test Cases' tab first.")
            else:
                valid_apps_for_review = [
                    app for app, cases in st.session_state.generated_test_cases.items()
                    if isinstance(cases, list) and cases and all(isinstance(tc, dict) for tc in cases)
                ]

                if not valid_apps_for_review:
                    st.warning("No applications with valid generated test cases available for review.")
                else:
                    # --- Application Selection for AI Review ---
                    if st.session_state.ai_review_selected_app not in valid_apps_for_review:
                        st.session_state.ai_review_selected_app = valid_apps_for_review[0] if valid_apps_for_review else None

                    selected_app_for_review = st.selectbox(
                        "Select Application to Review:",
                        options=valid_apps_for_review,
                        key="ai_review_selected_app", # Uses the session state key
                        help="Choose an application whose test cases you want the AI to review."
                    )

                    if selected_app_for_review:
                        st.markdown("---")
                        st.subheader(f"Reviewing Test Cases for: `{selected_app_for_review}`")

                        # Display summary of inputs to be used by AI
                        with st.expander("View Inputs for AI Review", expanded=False):
                            st.markdown("**Main Requirements Document (Summary):**")
                            st.text_area("Requirements Summary", st.session_state.extracted_text[:1000] + "..." if st.session_state.extracted_text else "N/A", height=150, disabled=True)

                            st.markdown("**Uploaded Context Files:**")
                            app_context_files = st.session_state.uploaded_context_files.get(selected_app_for_review, [])
                            if app_context_files:
                                for f in app_context_files:
                                    st.caption(f"- `{f.name}`")
                            else:
                                st.caption("_No context files were uploaded for this application._")

                        st.markdown("**Generated Test Cases:**")
                        if st.session_state.generated_test_cases:
                            st.markdown("---")
                            ui_components.display_test_cases_for_app(st.session_state.generated_test_cases, selected_app_for_review)
                        else:
                            st.caption("_No test cases generated for this application yet._")

                        # --- Start AI Review Button ---
                        if st.button(f"🚀 Start AI Review for `{selected_app_for_review}`", key="start_ai_review_btn", type="primary", disabled=st.session_state.ai_review_inprogress_flag):
                            utils.log_message(f"AI Review button clicked for app: {selected_app_for_review}", "INFO")
                            st.session_state.ai_review_inprogress_flag = True
                            st.session_state.ai_review_results_raw = None # Clear previous results
                            st.session_state.ai_review_suggestions_processed = None
                            st.session_state.ai_review_user_decisions = {}


                            creds_ok, creds_msg = llm_integration_core.check_credentials(
                                st.session_state.llm_provider, st.session_state.api_credentials,
                                st.session_state.openai_fallback_api_key, require_fallback_for_rag=True
                            )
                            if not creds_ok:
                                utils.log_message(f"AI Review failed: Credentials check failed - {creds_msg}", "ERROR")
                                st.error(f"Cannot start AI Review: {creds_msg}")
                                st.session_state.ai_review_inprogress_flag = False
                            elif not st.session_state.get("model_name"):
                                utils.log_message(f"AI Review failed: No model selected for {st.session_state.llm_provider}.", "ERROR")
                                st.error(f"Cannot start AI Review: No model selected for {st.session_state.llm_provider}.")
                                st.session_state.ai_review_inprogress_flag = False
                            else:
                                llm, embeddings = llm_integration_core.get_llm_and_embeddings(
                                    st.session_state.llm_provider, st.session_state.model_name,
                                    st.session_state.api_credentials, st.session_state.openai_fallback_api_key
                                )
                                if llm and embeddings:
                                    # Prepare context string
                                    additional_context_str = ""
                                    app_context_files_to_process = st.session_state.uploaded_context_files.get(selected_app_for_review, [])
                                    if app_context_files_to_process:
                                        for uploaded_file in app_context_files_to_process:
                                            try:
                                                extracted_content = file_processing.extract_text_from_file(uploaded_file)
                                                if extracted_content:
                                                    additional_context_str += f"\n\n--- Context from {uploaded_file.name} ---\n{extracted_content}\n--- End Context ---\n"
                                            except Exception as e:
                                                utils.log_message(f"Error processing context file {uploaded_file.name} for AI review: {e}", "WARNING")
                                                st.warning(f"Could not fully process context file: {uploaded_file.name}")

                                    current_tcs = st.session_state.generated_test_cases.get(selected_app_for_review, [])

                                    with st.spinner(f"AI is reviewing test cases for `{selected_app_for_review}`... This may take a moment."):
                                        review_output = llm_integration_core.perform_ai_test_case_review(
                                            main_requirements_text=st.session_state.extracted_text,
                                            additional_context_str=additional_context_str,
                                            existing_test_cases=current_tcs,
                                            llm=llm,
                                            embeddings=embeddings,
                                            provider_name=st.session_state.llm_provider,
                                            app_name=selected_app_for_review
                                        )
                                    st.session_state.ai_review_results_raw = review_output
                                    st.session_state.ai_review_inprogress_flag = False

                                    if review_output:
                                        utils.log_message("AI Review completed. Raw output received.", "INFO")
                                        st.session_state.ai_review_results_raw = review_output
                                        # Process raw results into a more structured format for UI
                                        # For now, this is a direct assignment, expecting the LLM to return proper  structure.
                                        # PreProcessing can be added here once it's done
                                        processed_suggestions = {
                                            "coverage_summary": review_output.get("coverage_summary", "Not provided."),
                                            "newly_suggested_test_cases": review_output.get("newly_suggested_test_cases", []),
                                            "modified_test_cases_suggestions": review_output.get("modified_test_cases_suggestions", []),
                                            "identified_duplicates": review_output.get("identified_duplicates", [])
                                        }
                                        st.session_state.ai_review_suggestions_processed = processed_suggestions
                                        utils.log_message(f"Processed AI review suggestions: {st.session_state.ai_review_suggestions_processed}", "DEBUG")

                                    else:
                                        utils.log_message("AI Review returned no output or failed.", "ERROR")
                                        st.error("AI Review process did not return any results. Check logs.")
                                        st.session_state.ai_review_results_raw = None # Ensure it's cleared on failure
                                        st.session_state.ai_review_suggestions_processed = None
                                    st.session_state.ai_review_inprogress_flag = False
                                    st.rerun()
                                else:
                                    utils.log_message("AI Review failed: LLM/Embeddings could not be initialized.", "ERROR")
                                    st.error("AI Review failed: LLM/Embeddings could not be initialized (RAG required). Check logs.")
                                    st.session_state.ai_review_inprogress_flag = False

                        # --- Display AI Review Results and Handle User Decisions ---
                        if st.session_state.ai_review_suggestions_processed and selected_app_for_review:
                            st.markdown("---")
                            st.subheader("AI Review Analysis & Suggestions")

                            # Display Coverage Summary
                            ui_components.render_ai_review_summary_display(
                                st.session_state.ai_review_suggestions_processed.get('coverage_summary')
                            )

                            ui_components.render_ai_suggestions(
                                st.session_state.ai_review_suggestions_processed,
                                selected_app_for_review
                            )

                            ui_components.render_apply_ai_review_changes_button(selected_app_for_review)

            # --- Apply AI Review Changes-
            if 'trigger_apply_ai_changes' in st.session_state and st.session_state.trigger_apply_ai_changes:
                app_to_update = st.session_state.trigger_apply_ai_changes
                utils.log_message(f"Attempting to apply AI review changes for app: {app_to_update}", "INFO")

                changes_applied_count = 0
                if app_to_update in st.session_state.generated_test_cases and \
                   st.session_state.ai_review_suggestions_processed and \
                   st.session_state.ai_review_user_decisions:

                    current_tcs_for_app = st.session_state.generated_test_cases[app_to_update]
                    if not isinstance(current_tcs_for_app, list): # Ensure it's a list
                        current_tcs_for_app = []
                        st.session_state.generated_test_cases[app_to_update] = current_tcs_for_app


                    # Determine the next available Test Case ID suffix
                    max_id_num = 0
                    for tc in current_tcs_for_app:
                        if isinstance(tc, dict) and "Test Case ID" in tc:
                            match = re.search(r'_(\d+)$', tc["Test Case ID"])
                            if match:
                                max_id_num = max(max_id_num, int(match.group(1)))
                    next_id_counter = max_id_num + 1

                    # Process accepted new test cases
                    new_suggestions = st.session_state.ai_review_suggestions_processed.get("newly_suggested_test_cases", [])
                    for i, suggested_tc_data in enumerate(new_suggestions):
                        suggestion_id = f"new_{app_to_update}_{i}"
                        if st.session_state.ai_review_user_decisions.get(suggestion_id) == "accept":
                            new_tc_to_add = dict(suggested_tc_data) # Make a copy
                            
                            # Assign a new unique Test Case ID
                            new_tc_to_add["Test Case ID"] = f"{app_to_update}_TC_{next_id_counter}"
                            next_id_counter += 1
                            
                            current_tcs_for_app.append(new_tc_to_add)
                            changes_applied_count += 1
                            utils.log_message(f"Applied new TC: {new_tc_to_add['Test Case ID']}", "INFO")
                    
                    new_tcs_applied_count = changes_applied_count # Store count of new TCs before processing modifications
                    modifications_applied_count = 0

                    # Process accepted modified test cases
                    modified_suggestions = st.session_state.ai_review_suggestions_processed.get("modified_test_cases_suggestions", [])
                    if modified_suggestions:
                        for mod_suggestion_details in modified_suggestions:
                            original_tc_id_to_modify = mod_suggestion_details.get("original_test_case_id")
                            suggestion_key = f"mod_{app_to_update}_{original_tc_id_to_modify}"

                            if st.session_state.ai_review_user_decisions.get(suggestion_key) == "accept":
                                suggested_data = mod_suggestion_details.get("suggested_test_case_data")
                                if not suggested_data or not isinstance(suggested_data, dict):
                                    utils.log_message(f"Skipping modification for TC ID '{original_tc_id_to_modify}': Suggested data is invalid.", "WARNING")
                                    continue

                                found_and_modified = False
                                for i, existing_tc in enumerate(current_tcs_for_app):
                                    if isinstance(existing_tc, dict) and existing_tc.get("Test Case ID") == original_tc_id_to_modify:
                                        # Ensure the Test Case ID from the original is preserved, even if AI changed it in suggested_data
                                        final_modified_data = dict(suggested_data)
                                        final_modified_data["Test Case ID"] = original_tc_id_to_modify
                                        
                                        current_tcs_for_app[i] = final_modified_data
                                        modifications_applied_count += 1
                                        found_and_modified = True
                                        utils.log_message(f"Applied modification for TC ID: {original_tc_id_to_modify}", "INFO")
                                        break
                                if not found_and_modified:
                                    utils.log_message(f"Could not find original TC ID '{original_tc_id_to_modify}' to apply modification.", "WARNING")
                    
                    changes_applied_count += modifications_applied_count # This variable now tracks total changes
                    duplicates_removed_count = 0

                    # Process accepted duplicate resolutions
                    duplicate_suggestions_groups = st.session_state.ai_review_suggestions_processed.get("identified_duplicates", [])
                    if duplicate_suggestions_groups:
                        tc_ids_to_remove_overall = set()
                        for dup_group_details in duplicate_suggestions_groups:
                            group_id = dup_group_details.get("duplicate_group_id")
                            tc_ids_in_group = dup_group_details.get("test_case_ids", [])
                            
                            if not group_id or not tc_ids_in_group:
                                continue

                            decision_key = f"dup_{app_to_update}_{group_id}"
                            tc_id_to_keep = st.session_state.ai_review_user_decisions.get(decision_key)

                            if tc_id_to_keep and tc_id_to_keep != "Resolve Later / No Action":
                                # Add all IDs in the group to removal set, except the one to keep
                                for tc_id_in_group_member in tc_ids_in_group:
                                    if tc_id_in_group_member != tc_id_to_keep:
                                        tc_ids_to_remove_overall.add(tc_id_in_group_member)
                                utils.log_message(f"Duplicate group '{group_id}': Keeping TC ID '{tc_id_to_keep}', marking others for removal: {tc_ids_in_group - {tc_id_to_keep}}", "INFO")
                        
                        if tc_ids_to_remove_overall:
                            original_len = len(current_tcs_for_app)
                            current_tcs_for_app[:] = [tc for tc in current_tcs_for_app if tc.get("Test Case ID") not in tc_ids_to_remove_overall]
                            duplicates_removed_count = original_len - len(current_tcs_for_app)
                            if duplicates_removed_count > 0:
                                 utils.log_message(f"Removed {duplicates_removed_count} duplicate test cases for app '{app_to_update}'.", "INFO")
                    

                    summary_messages = []
                    if new_tcs_applied_count > 0:
                        summary_messages.append(f"{new_tcs_applied_count} new test case(s) added")
                    if modifications_applied_count > 0:
                        summary_messages.append(f"{modifications_applied_count} existing test case(s) modified")
                    if duplicates_removed_count > 0:
                        summary_messages.append(f"{duplicates_removed_count} duplicate test case(s) removed")
                    
                    if not summary_messages:
                        st.info(f"No changes were accepted or applied for '{app_to_update}'.")
                    else:
                        st.success(f"For '{app_to_update}': {', '.join(summary_messages)}.")
                else:
                    st.warning(f"Could not apply changes for '{app_to_update}'. Necessary data not found in session state.")

                # Clear AI review states after attempting to apply
                st.session_state.ai_review_results_raw = None
                st.session_state.ai_review_suggestions_processed = None
                st.session_state.ai_review_user_decisions = {}
                if 'trigger_apply_ai_changes' in st.session_state: # Check before deleting
                    del st.session_state.trigger_apply_ai_changes # Consume the trigger
                
                st.rerun()


        # === Manual Refactor Tab (Original Tab 2, now Tab 3) ===
        with tab3: # Was tab2, now tab3
            st.header("Manual Refactor Generated Test Cases")

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
                    st.session_state.openai_fallback_api_key, require_fallback_for_rag=True
                )
                if not creds_ok:
                    utils.log_message(f"Single refactor failed: Credentials check failed - {creds_msg}", "ERROR")
                    st.error(f"Cannot refactor: {creds_msg}")
                elif not st.session_state.get("model_name"):
                    utils.log_message(f"Single refactor failed: No model selected for {st.session_state.llm_provider}.", "ERROR")
                    st.error(f"Cannot refactor: No model selected for {st.session_state.llm_provider}.")
                else:
                    llm, embeddings = llm_integration_core.get_llm_and_embeddings(
                        st.session_state.llm_provider, st.session_state.model_name,
                        st.session_state.api_credentials, st.session_state.openai_fallback_api_key
                    )
                    if llm and embeddings:
                        additional_context_str = ""
                        app_context_files_to_process = st.session_state.uploaded_context_files.get(req['app_name'], [])
                        if app_context_files_to_process:
                            for uploaded_file in app_context_files_to_process:
                                try:
                                    extracted_content = file_processing.extract_text_from_file(uploaded_file)
                                    if extracted_content:
                                        additional_context_str += f"\n\n--- Context from {uploaded_file.name} ---\n{extracted_content}\n--- End Context ---\n"
                                except Exception as e:
                                    utils.log_message(f"Error processing context file {uploaded_file.name} for single refactor: {e}", "WARNING")

                        with st.spinner(f"Refactoring Test Case '{req['tc_id']}'..."):
                            refactored_data = llm_integration_core.refactor_single_test_case(
                                req['app_name'], req['tc_id'], req['instructions'],
                                req['original_data'], llm, embeddings,
                                st.session_state.llm_provider,
                                st.session_state.extracted_text,
                                additional_context_str
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
                        utils.log_message("Single refactor failed: LLM/Embeddings could not be initialized.", "ERROR")
                        st.error("Refactoring failed: LLM/Embeddings could not be initialized (RAG required). Check logs.")
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
                    st.session_state.openai_fallback_api_key, require_fallback_for_rag=True
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
                    llm, embeddings = llm_integration_core.get_llm_and_embeddings(
                        st.session_state.llm_provider, st.session_state.model_name,
                        st.session_state.api_credentials, st.session_state.openai_fallback_api_key
                    )
                    if llm and embeddings:
                        additional_context_str = ""
                        app_context_files_to_process = st.session_state.uploaded_context_files.get(req['app_name'], [])
                        if app_context_files_to_process:
                            for uploaded_file in app_context_files_to_process:
                                try:
                                    extracted_content = file_processing.extract_text_from_file(uploaded_file)
                                    if extracted_content:
                                        additional_context_str += f"\n\n--- Context from {uploaded_file.name} ---\n{extracted_content}\n--- End Context ---\n"
                                except Exception as e:
                                    utils.log_message(f"Error processing context file {uploaded_file.name} for bulk refactor: {e}", "WARNING")

                        with st.spinner(f"Refactoring all test cases for '{req['app_name']}'..."):
                            # *** Call the NEW bulk refactoring function ***
                            # Ensure the function exists in llm_integration_core
                            if hasattr(llm_integration_core, 'refactor_all_test_cases'):
                                refactored_list = llm_integration_core.refactor_all_test_cases(
                                    req['app_name'], req['instructions'],
                                    req['original_data'], # Pass the list of original cases
                                    llm,
                                    embeddings,
                                    st.session_state.llm_provider,
                                    st.session_state.extracted_text,
                                    additional_context_str
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
                        utils.log_message("Bulk refactor failed: LLM/Embeddings could not be initialized.", "ERROR")
                        st.error("Refactoring failed: LLM/Embeddings could not be initialized (RAG required). Check logs.")
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
    st.info("⬅️ Upload one or more `.docx` or `.pdf` documents using the file uploader in the sidebar to begin.")
    # Updated caption to reflect context upload change
    st.caption("Optional context files (.txt, .md, .docx, .xlsx, .json, .yaml) can be uploaded per application in Step 3 after identifying applications.")

    st.caption("Note: Context files are not required for generation but can improve results.")
