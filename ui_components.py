# ui_components.py
"""Functions for rendering specific parts of the Streamlit UI."""

import streamlit as st
import pandas as pd
import os
from typing import Dict, Any, List, Optional

# Import config and utilities
# Make sure config.py and utils.py are in the same directory or accessible via PYTHONPATH
try:
    # Assuming config.py is accessible
    import config
    from config import (
        LLM_PROVIDER_CONFIG, FALLBACK_EMBEDDING_PROVIDERS, NO_CONTEXT_OPTION, # NO_CONTEXT_OPTION might be removed if sidebar changes
        EXCEL_EXPECTED_COLUMNS, APP_CONTEXT_FOLDER_PATH, APP_CONTEXT_FOLDER_NAME
    )
    # Assuming utils.py is in a 'helper' subfolder relative to this file
    # or accessible via PYTHONPATH
    from helper.utils import sanitize_filename, get_available_context_files, log_message # get_available_context_files might only be needed for sidebar now
except ImportError as e:
    # Basic logging if utils/config import fails
    def log_message(msg, level): print(f"{level}: {msg}")
    st.error(f"Failed to import required modules (config, helper.utils). Ensure they exist: {e}")
    # Optionally stop execution if core components are missing
    # st.stop()
except NameError as e:
     # Handle case where log_message itself failed to import
     def log_message(msg, level): print(f"{level}: {msg}")
     st.error(f"Failed to import required modules (config, helper.utils). Ensure they exist: {e}")


def render_llm_config_sidebar():
    """Renders the LLM Provider and Credential configuration in the sidebar."""
    st.header("⚙️ LLM Configuration")

    provider_options = list(LLM_PROVIDER_CONFIG.keys())

    # Ensure current provider selection is valid, default if not
    if 'llm_provider' not in st.session_state or st.session_state.llm_provider not in provider_options:
        st.session_state.llm_provider = provider_options[0] if provider_options else None

    # Callback function to update credentials in session state
    def update_credential(key_in_state, widget_key):
        if widget_key in st.session_state: # Ensure widget key exists before access
            st.session_state.api_credentials[key_in_state] = st.session_state[widget_key]
        else:
             log_message(f"Widget key '{widget_key}' not found in session state during callback for '{key_in_state}'.", "WARNING")


    # Callback function to update fallback key in session state
    def update_fallback_key():
        if 'openai_fallback_key_widget' in st.session_state: # Check key exists
            st.session_state.openai_fallback_api_key = st.session_state.openai_fallback_key_widget
        else:
            log_message("Widget key 'openai_fallback_key_widget' not found in session state during callback.", "WARNING")


    selected_provider = st.selectbox(
        "Select LLM Provider",
        options=provider_options,
        key="llm_provider", # Use session state key directly
        help="Choose the AI provider for generation and analysis."
    )

    # Handle case where no providers are configured
    if not selected_provider:
         st.warning("No LLM providers configured in `config.LLM_PROVIDER_CONFIG`.")
         return # Stop rendering config if no provider selected/available

    provider_config = LLM_PROVIDER_CONFIG[selected_provider]
    available_models = provider_config.get("models", [])

    # Ensure current model selection is valid for the provider, default if not
    current_model = st.session_state.get("model_name")
    if not available_models:
        st.warning(f"No models listed for {selected_provider} in configuration.")
        st.session_state.model_name = None
    # Check if current model is valid *for the selected provider*
    elif current_model not in available_models:
        st.session_state.model_name = available_models[0] # Default to first model

    # Render model selection only if models are available
    if available_models:
        st.selectbox(
            f"Select {selected_provider} Model",
            options=available_models,
            key="model_name", # Use session state key
            help=f"Choose a specific model from {selected_provider}."
        )

    # --- Credentials ---
    st.markdown("**API Credentials**")
    notes = provider_config.get("notes", "")
    if notes:
        st.caption(notes)

    required_creds = provider_config.get("credentials", [])
    if 'api_credentials' not in st.session_state:
        st.session_state.api_credentials = {}

    # Initialize missing credential keys in session state
    # Also, set default for Ollama base_url if needed
    for cred_key in required_creds:
        if cred_key not in st.session_state.api_credentials:
            # Set default for Ollama base_url
            if selected_provider == "Ollama" and cred_key == "base_url":
                st.session_state.api_credentials[cred_key] = "http://localhost:11434"
            else:
                st.session_state.api_credentials[cred_key] = ""

    # Render input fields for required credentials
    for cred_key in required_creds:
        # Skip rendering 'model' for Ollama as it's handled by dropdown above
        if selected_provider == "Ollama" and cred_key == "model":
            continue # Skip this credential key for Ollama

        widget_key = f"cred_{cred_key}_widget" # Unique key for the widget itself
        label = cred_key.replace("_", " ").title()
        is_secret = "key" in cred_key.lower() or "secret" in cred_key.lower() or "token" in cred_key.lower()
        input_type = "password" if is_secret else "default"

        # Special handling for Bedrock Embedding Model ID (Dropdown)
        if selected_provider == "AWS Bedrock" and cred_key == "embedding_model_id":
            bedrock_embed_models = provider_config.get("embedding_model_ids", [])
            if bedrock_embed_models:
                current_embed_model = st.session_state.api_credentials.get(cred_key, "")
                try:
                    if current_embed_model not in bedrock_embed_models:
                        current_embed_model = bedrock_embed_models[0] if bedrock_embed_models else ""
                        if st.session_state.api_credentials.get(cred_key) != current_embed_model:
                            st.session_state.api_credentials[cred_key] = current_embed_model
                    current_index = bedrock_embed_models.index(current_embed_model) if current_embed_model else 0
                except ValueError:
                    current_index = 0
                    if current_embed_model and bedrock_embed_models:
                        st.session_state.api_credentials[cred_key] = bedrock_embed_models[0]

                st.selectbox(
                    label,
                    options=bedrock_embed_models,
                    key=widget_key,
                    index=current_index,
                    help="Select the Bedrock Embedding Model ID enabled in your AWS account and region.",
                    on_change=update_credential, args=(cred_key, widget_key)
                )
            else: # Fallback to text input
                st.text_input(
                    label + " (Enter ID)",
                    type="default",
                    key=widget_key,
                    value=st.session_state.api_credentials.get(cred_key, ""),
                    help="Enter the Bedrock Embedding Model ID.",
                    on_change=update_credential, args=(cred_key, widget_key)
                )
        else: # Standard text input
            st.text_input(
                label,
                type=input_type,
                key=widget_key,
                value=st.session_state.api_credentials.get(cred_key, ""),
                help=f"Enter your {label}.",
                on_change=update_credential, args=(cred_key, widget_key)
            )

    # --- Fallback Key ---
    if selected_provider in FALLBACK_EMBEDDING_PROVIDERS:
        st.markdown("**OpenAI API Key (for RAG Fallback)**")
        st.caption(f"{selected_provider} requires OpenAI embeddings for RAG features.")
        if 'openai_fallback_api_key' not in st.session_state:
            st.session_state.openai_fallback_api_key = ""

        st.text_input(
            "OpenAI API Key",
            type="password",
            key="openai_fallback_key_widget",
            value=st.session_state.openai_fallback_api_key,
            help="Required only if using RAG features (like Generate) with this provider.",
            on_change=update_fallback_key
         )


def render_context_options_sidebar():
    """Renders information about the optional context folder in the sidebar."""
    # This function might become less relevant if context is only uploaded,
    # but can be kept for informational purposes or future use.
    st.subheader("🗂️ Optional: App Context Folder")
    st.caption(f"Previously, context files were loaded from `{APP_CONTEXT_FOLDER_NAME}`. "
               f"Context is now uploaded directly per application in Step 3.")

    # Optionally, still show if the folder exists and what's in it for reference
    if os.path.exists(APP_CONTEXT_FOLDER_PATH) and os.path.isdir(APP_CONTEXT_FOLDER_PATH):
        st.info(f"Legacy context folder found: `{APP_CONTEXT_FOLDER_PATH}` (Files here are no longer automatically used).")
        try:
            files = get_available_context_files() # Still uses the util function
            files_to_show = [f for f in files if f != NO_CONTEXT_OPTION]
            if files_to_show:
                st.write("Files detected (for reference only):")
                for f in files_to_show:
                     st.code(f)
            else:
                st.write("Legacy context folder is empty.")
        except Exception as e:
            st.warning(f"Could not list files in legacy context folder: {e}")
            log_message(f"Error listing legacy context files: {e}", "WARNING", exc_info=True)
    else:
        st.caption(f"Legacy context folder not found at: `{APP_CONTEXT_FOLDER_PATH}`.")


# *** MODIFIED FUNCTION ***
def render_app_context_selection(identified_apps: List[str]):
    """
    Renders the UI for selecting applications and uploading their context files.
    """
    st.markdown("---")
    st.subheader("3. Select Apps & Upload Context")

    if not identified_apps:
        st.info("Run 'Identify Applications' first to populate this section.")
        return

    # --- Application Selection ---
    st.write("**Select Applications to Generate Cases For:**")
    current_selection = st.session_state.get('selected_applications', [])
    valid_defaults = [app for app in current_selection if app in identified_apps]

    if not valid_defaults and identified_apps:
        valid_defaults = identified_apps # Default to all if none selected/valid

    selected_apps = st.multiselect(
        "Applications:",
        options=identified_apps,
        default=valid_defaults,
        label_visibility="collapsed",
        key="app_select_multiselect_widget"
    )

    # --- Context File Upload per Application ---
    if selected_apps:
        st.markdown("**Upload Context Files (Optional):**")
        st.caption("Upload relevant documents (.txt, .md, .docx, .xlsx, .json, .yaml) for each selected application.")

        # Initialize the new state variable if it doesn't exist
        if 'uploaded_context_files' not in st.session_state:
            st.session_state.uploaded_context_files = {}

        # Prune uploaded files state if app selection changes
        if set(selected_apps) != set(st.session_state.get('selected_applications', [])):
            current_uploaded = st.session_state.get('uploaded_context_files', {})
            st.session_state.uploaded_context_files = {
                app: files for app, files in current_uploaded.items() if app in selected_apps
            }
            # Update selected_applications state *after* pruning context based on the *new* selection
            st.session_state.selected_applications = selected_apps
            st.rerun() # Rerun to reflect pruning and new selection

        # Create columns for potentially better layout if many apps are selected
        num_columns = min(len(selected_apps), 3) # Max 3 columns
        cols = st.columns(num_columns)

        col_index = 0
        for app_name in selected_apps:
            with cols[col_index % num_columns]:
                st.markdown(f"**Context for `{app_name}`:**")
                widget_key = f"ctx_upload_{sanitize_filename(app_name)}_widget"
                allowed_types = ['txt', 'md', 'docx', 'xlsx', 'json', 'yaml']

                # Get currently uploaded files for this app from state
                # The file_uploader widget state persists automatically via its key
                # We store the result in our own state variable for downstream use
                uploaded_files_for_app = st.file_uploader(
                    f"Upload files for {app_name}",
                    type=allowed_types,
                    accept_multiple_files=True,
                    key=widget_key, # Unique key for this uploader instance
                    label_visibility="collapsed",
                    help=f"Upload context files ({', '.join(allowed_types)}) for {app_name}."
                )

                # Update our central state dictionary whenever the uploader changes
                # Check if the returned list is different from what's stored
                # Note: Comparing lists of UploadedFile objects directly might be tricky
                # A simpler approach is to just update the state on every run where files are present
                # Or compare based on file names and sizes if needed, but direct update is easier
                if uploaded_files_for_app is not None:
                     st.session_state.uploaded_context_files[app_name] = uploaded_files_for_app
                # Handle case where files are removed (uploader returns empty list)
                elif app_name in st.session_state.uploaded_context_files and not uploaded_files_for_app:
                     del st.session_state.uploaded_context_files[app_name]


                # Display names of currently uploaded files for this app
                current_files_in_state = st.session_state.uploaded_context_files.get(app_name, [])
                if current_files_in_state:
                    st.write(f"_{len(current_files_in_state)} file(s) staged:_")
                    for f in current_files_in_state:
                        st.caption(f"- `{f.name}` ({f.size} bytes)")
                else:
                    st.caption("_No context files uploaded._")

            col_index += 1 # Move to the next column

    elif identified_apps:
        st.info("Select one or more applications above to upload context files.")
# *** END MODIFIED FUNCTION ***


def display_results(test_cases_dict: Dict[str, Any]):
    """
    Displays the summary metrics and detailed results in expanders.
    (No changes needed in this function based on the request)
    """
    if not test_cases_dict:
        return

    st.subheader("📊 Results Summary")
    successful_apps = 0
    error_apps = 0
    total_cases_generated = 0
    app_names_with_results = list(test_cases_dict.keys())

    for app_name in app_names_with_results:
        cases_result = test_cases_dict.get(app_name)
        if isinstance(cases_result, list) and cases_result:
            if all(isinstance(item, dict) for item in cases_result):
                successful_apps += 1
                total_cases_generated += len(cases_result)
            else:
                error_apps += 1 # Treat list with non-dicts as an error
        else:
            error_apps += 1 # Error strings, empty lists, None, etc.

    col1, col2, col3 = st.columns(3)
    col1.metric("Applications Processed", len(app_names_with_results))
    col2.metric("Apps with Cases Generated", successful_apps, help=f"Total individual test cases generated: {total_cases_generated}")
    col3.metric(
        "Apps with Errors/No Cases",
        error_apps,
        delta=f"{error_apps} issues" if error_apps > 0 else "0",
        delta_color="inverse" if error_apps > 0 else "normal"
    )

    st.write("**Detailed Results per Application:**")
    for app_name in app_names_with_results:
        cases_result = test_cases_dict.get(app_name)
        # Use a unique key based on app_name for the expander
        expander_key = f"expander_{sanitize_filename(app_name)}"
        with st.expander(f"View Results for: {app_name}", expanded=False):
            if isinstance(cases_result, list) and cases_result:
                if all(isinstance(item, dict) for item in cases_result):
                    st.write(f"Generated {len(cases_result)} test cases:")
                    try:
                        df = pd.DataFrame(cases_result)
                        display_cols_present = []
                        other_cols_present = []
                        existing_cols_lower = {col.lower(): col for col in df.columns}

                        for col in EXCEL_EXPECTED_COLUMNS:
                            actual_col = existing_cols_lower.get(col.lower())
                            if actual_col:
                                display_cols_present.append(actual_col)
                            else:
                                df[col] = pd.NA
                                display_cols_present.append(col)

                        other_cols_present = [col for col in df.columns if col not in display_cols_present]
                        final_display_order = display_cols_present + other_cols_present
                        df_display = df[final_display_order]
                        st.dataframe(df_display, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not display results for '{app_name}' as a table: {e}")
                        st.json(cases_result) # Fallback to JSON view
                else:
                    st.warning(f"Data for '{app_name}' is a list but contains non-dictionary items.")
                    st.json(cases_result) # Show raw list
            elif isinstance(cases_result, str):
                if cases_result.lower().startswith("error"):
                    st.error(f"Error for {app_name}: {cases_result}")
                else:
                    st.warning(f"Status for {app_name}: {cases_result}")
            else:
                st.warning(f"No valid test cases or error message found for {app_name}.")

def display_test_cases_for_app(test_cases_dict: Dict[str, Any], target_app_name: str):
    """
    Displays the summary metrics and detailed results for a specific application,
    identified by target_app_name.
    """
    # Check if the main dictionary is empty
    if not test_cases_dict:
        st.warning("Test cases data is empty. Nothing to display.")
        return

    # Check if the target application exists in the dictionary
    if target_app_name not in test_cases_dict:
        st.warning(f"No data found for application: '{target_app_name}'. Cannot display results.")
        return

    # Get the results for the target application
    cases_result = test_cases_dict.get(target_app_name)

    st.write(f"**All Test Case Results for: {target_app_name}**")
    
    # Use a unique key based on app_name for the expander to avoid conflicts if this function
    # were to be called multiple times on the same page for different apps (though current design is for one).
    expander_key = f"expander_{sanitize_filename(target_app_name)}"

    # Default to expanded=True since we are focusing on this one app's details
    with st.expander(f"View Details for: {target_app_name}", expanded=True):
        if isinstance(cases_result, list) and cases_result:
            # This block handles the case where cases_result is a non-empty list.
            if all(isinstance(item, dict) for item in cases_result):
                # Successfully retrieved a list of test case dictionaries
                st.write(f"Generated {len(cases_result)} test cases:")
                try:
                    df = pd.DataFrame(cases_result)
                    display_cols_ordered = []
                    
                    # Create a mapping of lowercase column names to original column names
                    # This helps in case-insensitive matching of EXCEL_EXPECTED_COLUMNS
                    existing_cols_map = {col.lower(): col for col in df.columns}

                    # Add expected columns first, in the specified order
                    for expected_col in EXCEL_EXPECTED_COLUMNS:
                        actual_col_name = existing_cols_map.get(expected_col.lower())
                        if actual_col_name:
                            display_cols_ordered.append(actual_col_name)
                            # Remove from map to avoid re-adding it later
                            del existing_cols_map[expected_col.lower()] 
                        else:
                            # If an expected column is missing from the data, add it with NA values
                            df[expected_col] = pd.NA 
                            display_cols_ordered.append(expected_col)
                    
                    # Add any other columns from the DataFrame that were not in EXCEL_EXPECTED_COLUMNS
                    # These are the remaining columns from existing_cols_map (values are original column names)
                    other_cols_present = list(existing_cols_map.values())
                    final_display_order = display_cols_ordered + other_cols_present
                    
                    # Display the DataFrame with the reordered columns
                    df_display = df[final_display_order]
                    st.dataframe(df_display, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not display results for '{target_app_name}' as a table: {e}")
                    st.write("Raw data:")
                    st.json(cases_result) # Fallback to JSON view if DataFrame conversion fails
            else:
                # Data is a list but contains items that are not dictionaries
                st.warning(f"Data for '{target_app_name}' is a list but contains non-dictionary items.")
                st.write("Raw data:")
                st.json(cases_result) # Show raw list
        elif isinstance(cases_result, str):
            # Handle cases where cases_result is a string (e.g., an error message or a status update)
            if "error" in cases_result.lower(): # General check for "error" in the string
                st.error(f"Error recorded for {target_app_name}: {cases_result}")
            else:
                st.info(f"Status for {target_app_name}: {cases_result}") # e.g., "No applicable scenarios found"
        elif cases_result is None or (isinstance(cases_result, list) and not cases_result): 
            # Handles None or empty list explicitly, indicating no test cases
            st.info(f"No test cases were generated or are available for {target_app_name}.")
        else:
            # Catch-all for other unexpected data types for cases_result
            st.warning(f"Data for '{target_app_name}' is in an unexpected format (Type: {type(cases_result).__name__}).")
            st.write("Raw data:")
            st.json(cases_result) # Display the raw data if its format is not recognized

# --- Refactoring UI ---

def render_modification_confirmation_ui():
    """Renders the UI to display and confirm or discard the refactored test cases."""
    # Check if there's a bulk refactor request and proposed data
    if 'refactor_all_request' in st.session_state and st.session_state.refactor_all_request:
        refactor_request = st.session_state.refactor_all_request
        app_name = refactor_request["app_name"]
        instructions = refactor_request["instructions"]
        refactored_data = st.session_state.get('refactored_test_cases', None) # Get proposed changes

        st.info(f"Refactoring all test cases for application `{app_name}` with the following instructions:\n\n`{instructions}`")

        if refactored_data:
            st.subheader("Proposed Refactored Test Cases:")
            try:
                df = pd.DataFrame(refactored_data)
                # Attempt to display in a structured way 
                display_cols_present = []
                other_cols_present = []
                existing_cols_lower = {col.lower(): col for col in df.columns}

                for col in EXCEL_EXPECTED_COLUMNS:
                    actual_col = existing_cols_lower.get(col.lower())
                    if actual_col:
                        display_cols_present.append(actual_col)
                    else:
                        df[col] = pd.NA # Add missing expected columns if needed
                        display_cols_present.append(col)

                other_cols_present = [col for col in df.columns if col not in display_cols_present]
                final_display_order = display_cols_present + other_cols_present
                df_display = df[final_display_order]
                st.dataframe(df_display, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display refactored test cases as a table: {e}")
                st.json(refactored_data) # Fallback to JSON

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✅ Apply These Changes", key="apply_all_btn", type="primary"):
                    st.session_state.generated_test_cases[app_name] = refactored_data
                    st.success(f"All test cases for application '{app_name}' have been updated.")
                    # Clear modification state
                    st.session_state.refactor_all_request = None
                    st.session_state.refactored_test_cases = None
                    st.rerun()
            with col2:
                if st.button("❌ Discard", key="discard_all_btn"):
                    st.info("Refactoring discarded.")
                    # Clear modification state
                    st.session_state.refactor_all_request = None
                    st.session_state.refactored_test_cases = None
                    st.rerun()
        else:
            # This state occurs after clicking "Get Refactored Versions" but before LLM responds
            st.info("Waiting for refactored test cases to be generated...")

    # Check if there's a single modification request (original functionality)
    elif 'modification_app_name' in st.session_state and st.session_state.modification_app_name:
        required_keys = ['modification_app_name', 'modification_tc_id', 'proposed_modification_data', 'original_tc_data_for_diff']
        if not all(k in st.session_state and st.session_state[k] is not None for k in required_keys):
            # Don't show error if it's just waiting for bulk refactor
            if not ('refactor_all_request' in st.session_state and st.session_state.refactor_all_request):
                 st.error("Single modification state is incomplete. Cannot render confirmation.")
            # Clear potentially partial state
            for k in required_keys: st.session_state[k] = None
            st.rerun()
            return

        prop_app = st.session_state.modification_app_name
        prop_id = st.session_state.modification_tc_id
        prop_data = st.session_state.proposed_modification_data
        orig_data = st.session_state.original_tc_data_for_diff

        st.info("Proposed single modification ready for review:")
        st.write(f"**Application:** `{prop_app}` | **Test Case ID:** `{prop_id}`")

        col_o, col_p = st.columns(2)
        with col_o:
            st.write("**Original Test Case:**")
            st.json(orig_data, expanded=False)
        with col_p:
            st.write("**Proposed Test Case:**")
            st.json(prop_data, expanded=False)

        discard_btn_col, apply_btn_col = st.columns([1, 1])
        with discard_btn_col:
            if st.button("❌ Discard Change", key="discard_mod_btn"):
                # Clear modification state
                for k in required_keys: st.session_state[k] = None
                st.success("Modification discarded.")
                st.rerun()

        with apply_btn_col:
            if st.button("✅ Apply Change", key="apply_mod_btn", type="primary"):
                if 'generated_test_cases' not in st.session_state or prop_app not in st.session_state.generated_test_cases:
                    st.error(f"Cannot apply change: Results for application '{prop_app}' not found in session state.")
                    for k in required_keys: st.session_state[k] = None
                    st.rerun()
                    return

                app_cases = st.session_state.generated_test_cases.get(prop_app)
                applied = False
                apply_error = None # Variable to store specific error during apply
                if isinstance(app_cases, list):
                    for i, tc in enumerate(app_cases):
                        if isinstance(tc, dict) and tc.get("Test Case ID") == prop_id:
                            try:
                                st.session_state.generated_test_cases[prop_app][i] = prop_data
                                st.success(f"Test Case '{prop_id}' in application '{prop_app}' has been updated.")
                                applied = True
                                break
                            except IndexError as idx_err:
                                apply_error = f"Internal Error: Index {i} out of bounds for '{prop_app}' results. {idx_err}"
                                break
                            except Exception as e:
                                apply_error = f"Failed to apply change in session state: {e}"
                                break # Stop trying on error

                if not applied:
                     if apply_error:
                          st.error(apply_error)
                          log_message(apply_error, "ERROR")
                     else:
                          error_msg = f"Could not find Test Case ID '{prop_id}' within the generated results for application '{prop_app}'."
                          st.error(error_msg)
                          log_message(error_msg, "ERROR")

                if applied:
                    for k in required_keys: st.session_state[k] = None
                    st.rerun()
    # else: No active modification request (neither single nor bulk)


def render_modification_request_ui():
    """Renders the UI to select an application and provide refactoring instructions for all its test cases."""
    st.subheader("✍️ Refactor All Test Cases") # Changed subheader

    valid_apps_for_mod = [
        app for app, cases in st.session_state.get('generated_test_cases', {}).items()
        if isinstance(cases, list) and cases and all(isinstance(tc, dict) for tc in cases)
    ]

    if not valid_apps_for_mod:
        st.info("No test cases with valid data are available to refactor. Please generate results first.")
        return

    sel_app_mod = st.selectbox(
        "Select Application:",
        options=valid_apps_for_mod,
        key="mod_app_select_widget",
        help="Choose the application containing the test cases to refactor." # Updated help text
    )

    # Removed individual test case selection logic

    mod_instructions = st.text_area(
        f"Refactoring Instructions for ALL Test Cases in `{sel_app_mod if sel_app_mod else 'selected application'}`:", # Updated label
        key="mod_instructions_widget",
        height=150,
        placeholder="e.g., 'Change the priority to High', 'Add a step to verify the confirmation email', 'Rewrite expected results for clarity'",
        help="Clearly describe the changes you want the AI to make to ALL test cases in this application." # Updated help text
    )

    button_disabled = not (sel_app_mod and mod_instructions) # Updated button disable logic
    if st.button("🚀 Get Refactored Versions for All Cases", key="get_refactor_btn", disabled=button_disabled): # Updated button text
        app_test_cases = st.session_state.generated_test_cases.get(sel_app_mod, [])
        # Store request for bulk refactoring
        st.session_state.refactor_all_request = {
            "app_name": sel_app_mod,
            "instructions": mod_instructions,
            "original_data": app_test_cases # Store the list of original cases
        }
        # Clear any previous single refactor state (if exists)
        st.session_state.refactor_request = None
        st.session_state.refactored_test_cases = None # Clear previous bulk results
        st.rerun()

    if button_disabled: # Updated caption logic
        st.caption("Please select an application and enter refactoring instructions.")

# --- NEW UI Components for AI Review Tab ---

def display_test_cases_summary(test_cases: List[Dict], max_display: int = 5):
    """Displays a compact summary of test cases, e.g., for the AI Review input section."""
    if not test_cases:
        st.caption("_No test cases to display._")
        return

    st.caption(f"Showing first {min(len(test_cases), max_display)} of {len(test_cases)} test cases:")
    for i, tc in enumerate(test_cases):
        if i >= max_display:
            break
        tc_id = tc.get("Test Case ID", f"Unknown ID {i+1}")
        tc_name = tc.get("Test Case Name", "Unnamed Test Case")
        st.markdown(f"- **{tc_id}:** {tc_name}")

def render_ai_review_summary_display(summary_text: Optional[str]):
    """Displays the AI's coverage summary."""
    st.subheader("AI Coverage Summary")
    if summary_text:
        st.markdown(summary_text)
    else:
        st.info("Coverage summary not provided by AI or not yet available.")

def render_ai_suggestions(
    suggestions_processed: Optional[Dict[str, List[Dict]]],
    app_name: str
):
    """
    Renders the processed AI suggestions for new test cases, modifications, and duplicates.
    This is a placeholder and will need to be broken down into more specific rendering functions
    and integrate with st.session_state.ai_review_user_decisions.
    """
    if not suggestions_processed:
        st.info("No AI suggestions processed or available yet.")
        return

    # Placeholder: Displaying raw processed suggestions for now
    # In a full implementation, each section (new, modified, duplicates) would have
    # its own rendering logic with interactive elements (checkboxes, buttons)
    # to update st.session_state.ai_review_user_decisions.

    new_suggestions = suggestions_processed.get("newly_suggested_test_cases", [])
    modified_suggestions = suggestions_processed.get("modified_test_cases_suggestions", [])
    duplicate_suggestions = suggestions_processed.get("identified_duplicates", [])

    if not new_suggestions and not modified_suggestions and not duplicate_suggestions:
        st.info("AI analysis complete, but no specific suggestions were provided in the expected format.")
        return

    # Initialize user decisions if not present
    if 'ai_review_user_decisions' not in st.session_state:
        st.session_state.ai_review_user_decisions = {}

    # --- Render New Test Case Suggestions ---
    if new_suggestions:
        st.subheader(f"New Test Case Suggestions ({len(new_suggestions)})")
        for i, new_tc in enumerate(new_suggestions):
            suggestion_id = f"new_{app_name}_{i}"
            with st.expander(f"New Suggestion {i+1}: {new_tc.get('Test Case Name', 'Unnamed New TC')}", expanded=False):
                st.dataframe(pd.DataFrame([new_tc])) # Display TC details in a table

                # Ensure EXCEL_EXPECTED_COLUMNS is available
                cols_to_check = config.EXCEL_EXPECTED_COLUMNS if hasattr(config, 'EXCEL_EXPECTED_COLUMNS') else list(new_tc.keys())
                missing_fields = [field for field in cols_to_check if field not in new_tc or not new_tc.get(field)]
                if missing_fields:
                    st.warning(f"AI Suggestion Missing Fields: {', '.join(missing_fields)}")


                decision = st.radio(
                    "Accept this new test case?",
                    options=["No Decision", "Accept", "Reject"],
                    key=f"decision_{suggestion_id}",
                    horizontal=True,
                    index=0 # Default to "No Decision"
                )
                if decision != "No Decision":
                    st.session_state.ai_review_user_decisions[suggestion_id] = decision.lower()
                elif suggestion_id in st.session_state.ai_review_user_decisions and decision == "No Decision":
                    # Allow user to revert to "No Decision"
                    del st.session_state.ai_review_user_decisions[suggestion_id]

        log_message(f"UI: Displaying {len(new_suggestions)} new TC suggestions for app '{app_name}'.", "DEBUG")

    # --- Render Modified Test Case Suggestions ---
    if modified_suggestions:
        st.subheader(f"Suggestions for Modifying Existing Test Cases ({len(modified_suggestions)})")
        for i, mod_suggestion in enumerate(modified_suggestions):
            original_tc_id = mod_suggestion.get("original_test_case_id")
            reason = mod_suggestion.get("modification_reason", "No reason provided.")
            suggested_tc_data = mod_suggestion.get("suggested_test_case_data")

            if not original_tc_id or not suggested_tc_data:
                st.warning(f"Skipping modification suggestion {i+1} due to missing original ID or suggested data.")
                continue

            suggestion_id_key = f"mod_{app_name}_{original_tc_id}"
            
            # Find the original test case
            original_tc = None
            if app_name in st.session_state.get('generated_test_cases', {}) and \
               isinstance(st.session_state.generated_test_cases[app_name], list):
                for tc in st.session_state.generated_test_cases[app_name]:
                    if isinstance(tc, dict) and tc.get("Test Case ID") == original_tc_id:
                        original_tc = tc
                        break
            
            with st.expander(f"Modification for TC ID '{original_tc_id}': {suggested_tc_data.get('Test Case Name', 'N/A')}", expanded=False):
                st.markdown(f"**Reason for Suggestion:** {reason}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Test Case:**")
                    if original_tc:
                        st.json(original_tc, expanded=False)
                    else:
                        st.warning(f"Could not find original test case with ID: {original_tc_id}")
                
                with col2:
                    st.markdown("**AI's Suggested Version:**")
                    st.json(suggested_tc_data, expanded=False)
                    # Check for missing fields in AI suggestion
                    cols_to_check = config.EXCEL_EXPECTED_COLUMNS if hasattr(config, 'EXCEL_EXPECTED_COLUMNS') else list(suggested_tc_data.keys())
                    missing_fields_suggested = [field for field in cols_to_check if field not in suggested_tc_data or not suggested_tc_data.get(field)]
                    if missing_fields_suggested:
                        st.warning(f"AI's Suggested Version Missing Fields: {', '.join(missing_fields_suggested)}")


                current_decision = st.session_state.ai_review_user_decisions.get(suggestion_id_key, "No Decision")
                options = ["No Decision", "Accept", "Reject"]
                try:
                    current_index = options.index(current_decision.title()) # Ensure title case for matching
                except ValueError:
                    current_index = 0 # Default to "No Decision"

                decision = st.radio(
                    "Your decision for this modification:",
                    options=options,
                    key=f"decision_{suggestion_id_key}",
                    horizontal=True,
                    index=current_index
                )
                if decision != "No Decision":
                    st.session_state.ai_review_user_decisions[suggestion_id_key] = decision.lower()
                elif suggestion_id_key in st.session_state.ai_review_user_decisions and decision == "No Decision":
                     del st.session_state.ai_review_user_decisions[suggestion_id_key]
        log_message(f"UI: Displaying {len(modified_suggestions)} modification suggestions for app '{app_name}'.", "DEBUG")

    # --- Render Identified Duplicate Test Cases ---
    if duplicate_suggestions:
        st.subheader(f"Identified Duplicate Test Cases ({len(duplicate_suggestions)} group(s))")
        for i, dup_group in enumerate(duplicate_suggestions):
            group_id = dup_group.get("duplicate_group_id", f"group_{i}")
            tc_ids_in_group = dup_group.get("test_case_ids", [])
            reason = dup_group.get("reason", "No reason provided.")

            if not tc_ids_in_group:
                st.warning(f"Skipping duplicate group {group_id} as it contains no test case IDs.")
                continue

            suggestion_id_key = f"dup_{app_name}_{group_id}"
            
            with st.expander(f"Duplicate Group '{group_id}': ({len(tc_ids_in_group)} TCs)", expanded=False):
                st.markdown(f"**Reason for flagging as duplicates:** {reason}")
                st.markdown("**Test Cases in this group:**")

                radio_options = ["Resolve Later / No Action"]
                radio_captions = ["Do nothing for this group now."]
                
                original_tcs_in_app = st.session_state.get('generated_test_cases', {}).get(app_name, [])

                for tc_id in tc_ids_in_group:
                    tc_summary = tc_id # Default to ID if not found
                    for tc_data in original_tcs_in_app:
                        if isinstance(tc_data, dict) and tc_data.get("Test Case ID") == tc_id:
                            tc_summary = f"{tc_id} - {tc_data.get('Test Case Name', 'Unnamed TC')}"
                            break
                    radio_options.append(tc_id)
                    radio_captions.append(f"Keep this Test Case: {tc_summary}")

                current_decision_for_group = st.session_state.ai_review_user_decisions.get(suggestion_id_key, "Resolve Later / No Action")
                
                try:
                    # Find index of current decision. If current_decision is a TC ID, it will be in radio_options.
                    current_index = radio_options.index(current_decision_for_group)
                except ValueError:
                    current_index = 0 # Default to "Resolve Later / No Action"

                chosen_tc_to_keep = st.radio(
                    "Which Test Case to keep from this group? (Others will be marked for removal)",
                    options=radio_options,
                    captions=radio_captions,
                    key=f"decision_{suggestion_id_key}",
                    index=current_index
                )

                if chosen_tc_to_keep != "Resolve Later / No Action":
                    st.session_state.ai_review_user_decisions[suggestion_id_key] = chosen_tc_to_keep # Store the ID of the TC to keep
                elif suggestion_id_key in st.session_state.ai_review_user_decisions and chosen_tc_to_keep == "Resolve Later / No Action":
                    del st.session_state.ai_review_user_decisions[suggestion_id_key]
        log_message(f"UI: Displaying {len(duplicate_suggestions)} duplicate groups for app '{app_name}'.", "DEBUG")


def render_apply_ai_review_changes_button(app_name: str):
    """
    Renders the button to apply accepted AI review changes.
    The actual logic for applying changes will reside in main_app.py.
    """
    # This key needs to be unique and handled in main_app.py
    if st.button(f"✅ Apply Accepted AI Changes for `{app_name}`", key=f"apply_ai_changes_{app_name}_btn", type="primary"):
        # Set a flag or trigger in session state that main_app.py can react to
        st.session_state.trigger_apply_ai_changes = app_name
        # No direct action here; main_app.py will handle it and rerun.
        log_message(f"UI: 'Apply AI Changes' button clicked for app '{app_name}'. Flag set.", "INFO")
        st.rerun() # Rerun to allow main_app.py to process the flag
