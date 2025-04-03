# ui_components.py
"""Functions for rendering specific parts of the Streamlit UI."""

import streamlit as st
import pandas as pd
import os
from typing import Dict, Any, List, Optional

# Import config and utilities
# Make sure config.py and utils.py are in the same directory or accessible via PYTHONPATH
try:
    from config import (
        LLM_PROVIDER_CONFIG, FALLBACK_EMBEDDING_PROVIDERS, NO_CONTEXT_OPTION,
        EXCEL_EXPECTED_COLUMNS, APP_CONTEXT_FOLDER_PATH, APP_CONTEXT_FOLDER_NAME
    )
    from utils import sanitize_filename, get_available_context_files
except ImportError as e:
    st.error(f"Failed to import required modules (config, utils). Ensure they exist: {e}")
    # Optionally stop execution if core components are missing
    # st.stop()

def render_llm_config_sidebar():
    """Renders the LLM Provider and Credential configuration in the sidebar."""
    st.header("⚙️ LLM Configuration")

    provider_options = list(LLM_PROVIDER_CONFIG.keys())

    # Ensure current provider selection is valid, default if not
    if 'llm_provider' not in st.session_state or st.session_state.llm_provider not in provider_options:
        st.session_state.llm_provider = provider_options[0]

    # Callback function to update credentials in session state
    def update_credential(key_in_state, widget_key):
        st.session_state.api_credentials[key_in_state] = st.session_state[widget_key]

    # Callback function to update fallback key in session state
    def update_fallback_key():
        st.session_state.openai_fallback_api_key = st.session_state.openai_fallback_key_widget

    selected_provider = st.selectbox(
        "Select LLM Provider",
        options=provider_options,
        key="llm_provider", # Use session state key directly
        help="Choose the AI provider for generation and analysis."
    )

    provider_config = LLM_PROVIDER_CONFIG[selected_provider]
    available_models = provider_config.get("models", [])

    # Ensure current model selection is valid for the provider, default if not
    current_model = st.session_state.get("model_name")
    if not available_models:
        st.warning(f"No models listed for {selected_provider} in configuration.")
        st.session_state.model_name = None
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
    for cred_key in required_creds:
        if cred_key not in st.session_state.api_credentials:
            st.session_state.api_credentials[cred_key] = ""

    # Render input fields for required credentials
    for cred_key in required_creds:
        widget_key = f"cred_{cred_key}_widget" # Unique key for the widget itself
        label = cred_key.replace("_", " ").title()
        is_secret = "key" in cred_key.lower() or "secret" in cred_key.lower()
        input_type = "password" if is_secret else "default"

        # Special handling for Bedrock Embedding Model ID (Dropdown)
        if selected_provider == "AWS Bedrock" and cred_key == "embedding_model_id":
            bedrock_embed_models = provider_config.get("embedding_model_ids", [])
            if bedrock_embed_models:
                current_embed_model = st.session_state.api_credentials.get(cred_key, "")
                 # Find index, default to 0 if not found or empty
                try:
                    # Ensure the default value is valid before finding index
                    if current_embed_model not in bedrock_embed_models:
                         current_embed_model = bedrock_embed_models[0] if bedrock_embed_models else ""
                         # Update state immediately if correction was needed
                         if st.session_state.api_credentials.get(cred_key) != current_embed_model:
                              st.session_state.api_credentials[cred_key] = current_embed_model

                    current_index = bedrock_embed_models.index(current_embed_model) if current_embed_model else 0

                except ValueError:
                    current_index = 0
                    # Update state if current value was invalid
                    if current_embed_model and bedrock_embed_models:
                         st.session_state.api_credentials[cred_key] = bedrock_embed_models[0]

                st.selectbox(
                    label,
                    options=bedrock_embed_models,
                    key=widget_key, # Unique key for widget
                    index=current_index,
                    help="Select the Bedrock Embedding Model ID enabled in your AWS account and region.",
                    # Use args to pass parameters to the callback
                    on_change=update_credential, args=(cred_key, widget_key)
                )
            else: # Fallback to text input if no models listed in config
                st.text_input(
                    label + " (Enter ID)",
                    type="default",
                    key=widget_key,
                    value=st.session_state.api_credentials.get(cred_key, ""),
                    help="Enter the Bedrock Embedding Model ID.",
                     on_change=update_credential, args=(cred_key, widget_key)
                )
        else: # Standard text input for other credentials
            st.text_input(
                label,
                type=input_type,
                key=widget_key, # Unique widget key
                value=st.session_state.api_credentials.get(cred_key, ""), # Get value from state
                help=f"Enter your {label}.",
                # Update session state when the input changes
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
            key="openai_fallback_key_widget", # Widget key
            value=st.session_state.openai_fallback_api_key, # Get value from state
            help="Required only if using RAG features (like Generate) with this provider.",
            # Update session state on change
            on_change=update_fallback_key
         )


def render_context_options_sidebar():
    """Renders information about the optional context folder in the sidebar."""
    st.subheader("🗂️ Optional: App Context")
    # Use APP_CONTEXT_FOLDER_NAME for display consistency
    st.caption(f"Place `.yaml` or `.yml` files containing additional context "
               f"(e.g., API specs, data models) in a folder named "
               f"`{APP_CONTEXT_FOLDER_NAME}` in the **same directory where you run the Streamlit script** "
               f"(`{os.getcwd()}`).") # Show current working directory for clarity

    # Use APP_CONTEXT_FOLDER_PATH for checking existence
    if os.path.exists(APP_CONTEXT_FOLDER_PATH) and os.path.isdir(APP_CONTEXT_FOLDER_PATH):
        st.success(f"Context folder found: `{APP_CONTEXT_FOLDER_PATH}`")
        try:
             files = get_available_context_files()
             if len(files) > 1: # More than just "None"
                 st.write("Detected context files (base names):")
                 # Use columns for better display if many files
                 files_to_show = [f for f in files if f != NO_CONTEXT_OPTION]
                 cols = st.columns(3)
                 for i, f in enumerate(files_to_show):
                      with cols[i % 3]:
                           st.code(f)

             else:
                 st.info("Context folder found, but it contains no `.yaml`/`.yml` files.")
        except Exception as e:
             st.warning(f"Could not list context files: {e}")
    else:
        st.warning(f"Context folder not found at: `{APP_CONTEXT_FOLDER_PATH}`. Create it to add context files.")


def render_app_context_selection(identified_apps: List[str]):
    """Renders the UI for selecting applications and their context files."""
    st.markdown("---")
    st.subheader("3. Select Apps & Context")

    if not identified_apps:
        st.info("Run 'Identify Applications' first to populate this section.")
        return

    col1, col2 = st.columns([0.6, 0.4]) # App selection wider than context

    with col1:
        st.write("**Select Applications to Generate Cases For:**")
        # Ensure selections are valid options from identified_apps
        current_selection = st.session_state.get('selected_applications', [])
        valid_defaults = [app for app in current_selection if app in identified_apps]

        # Default to all identified apps if no valid selection exists or selection is empty
        if not valid_defaults and identified_apps:
            valid_defaults = identified_apps

        selected_apps = st.multiselect(
            "Applications:",
            options=identified_apps,
            default=valid_defaults,
            label_visibility="collapsed",
            key="app_select_multiselect_widget" # Unique widget key
        )
        # Update session state if selection changes
        if set(selected_apps) != set(st.session_state.get('selected_applications', [])):
            st.session_state.selected_applications = selected_apps
            # Prune context selections for apps that are no longer selected
            current_ctx = st.session_state.get('context_file_selections', {})
            st.session_state.context_file_selections = {
                app: ctx for app, ctx in current_ctx.items() if app in selected_apps
            }
            st.rerun() # Rerun to update context selection UI immediately

    with col2:
        if selected_apps:
            st.write("**Select Context File (Optional):**")
            # Get available context files ONCE per render of this section
            available_context_options = st.session_state.get('available_context_files', [NO_CONTEXT_OPTION])
            if not available_context_options: # Ensure 'None' is always an option
                 available_context_options = [NO_CONTEXT_OPTION]

            context_changed = False
            # Use a temporary dict to collect changes before updating state
            updated_contexts = st.session_state.get('context_file_selections', {}).copy()

            for app_name in selected_apps:
                widget_key = f"ctx_sel_{sanitize_filename(app_name)}_widget"
                current_app_context = updated_contexts.get(app_name, NO_CONTEXT_OPTION)

                # Ensure the current selection is valid, default to None if not
                if current_app_context not in available_context_options:
                    current_app_context = NO_CONTEXT_OPTION
                    # Update the temporary dict if correction was needed
                    if updated_contexts.get(app_name) != NO_CONTEXT_OPTION:
                         updated_contexts[app_name] = NO_CONTEXT_OPTION
                         context_changed = True # Mark change if correction happened

                try:
                    current_index = available_context_options.index(current_app_context)
                except ValueError:
                    current_index = 0 # Default to 'None' index

                # Callback to update the temporary dictionary
                def update_context_dict(app, key):
                     updated_contexts[app] = st.session_state[key]
                     # Mark that a change occurred
                     st.session_state['context_selection_made'] = True # Use a flag

                selected_context = st.selectbox(
                    f"Context for '{app_name}':",
                    options=available_context_options,
                    index=current_index,
                    key=widget_key,
                    help=f"Select a YAML context file relevant to {app_name}.",
                    on_change=update_context_dict, args=(app_name, widget_key)
                )

            # After rendering all selectboxes, check the flag and update the main session state if needed
            if st.session_state.get('context_selection_made', False):
                 st.session_state.context_file_selections = updated_contexts
                 st.session_state['context_selection_made'] = False # Reset flag
                 # No rerun needed here, state is updated for next action

        elif identified_apps:
             st.info("Select one or more applications from the left.")


def display_results(test_cases_dict: Dict[str, Any]):
    """
    Displays the summary metrics and detailed results in expanders.

    Args:
        test_cases_dict: The dictionary containing results per application.
                         Values can be lists of dicts (success) or error strings.
    """
    if not test_cases_dict:
        # Don't show anything if empty, main app shows initial message
        # st.info("No results generated yet.")
        return

    st.subheader("📊 Results Summary")
    successful_apps = 0
    error_apps = 0
    total_cases_generated = 0
    app_names_with_results = list(test_cases_dict.keys())

    for app_name in app_names_with_results:
        cases_result = test_cases_dict.get(app_name) # Use .get for safety
        if isinstance(cases_result, list) and cases_result:
            # Further check if list items are dicts (basic validation)
            if all(isinstance(item, dict) for item in cases_result):
                successful_apps += 1
                total_cases_generated += len(cases_result)
            else:
                 error_apps += 1 # Treat list with non-dicts as an error case
        else:
            error_apps += 1 # Includes error strings, empty lists, None, etc.

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
        # *** CORRECTED LINE: Removed the 'key' argument ***
        with st.expander(f"View Results for: {app_name}", expanded=False):
            if isinstance(cases_result, list) and cases_result:
                 # Check again if items are dicts before trying DataFrame
                 if all(isinstance(item, dict) for item in cases_result):
                    st.write(f"Generated {len(cases_result)} test cases:")
                    try:
                        # Use standardized expected columns from config
                        df = pd.DataFrame(cases_result)
                        display_cols_present = []
                        other_cols_present = []

                        # Ensure expected columns are present and ordered first
                        existing_cols_lower = {col.lower(): col for col in df.columns}
                        for col in EXCEL_EXPECTED_COLUMNS:
                             actual_col = existing_cols_lower.get(col.lower())
                             if actual_col:
                                  display_cols_present.append(actual_col)
                             else:
                                  df[col] = pd.NA # Add missing expected columns
                                  display_cols_present.append(col)

                        # Add any extra columns found at the end
                        other_cols_present = [col for col in df.columns if col not in display_cols_present]

                        # Select columns in desired order for display
                        # Use original casing from df for selection, then potentially rename headers later if needed
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
                # Display error or warning string
                if cases_result.lower().startswith("error"):
                     st.error(f"Error for {app_name}: {cases_result}")
                else:
                     st.warning(f"Status for {app_name}: {cases_result}")
            else:
                st.warning(f"No valid test cases or error message found for {app_name}.")

# --- Refactoring UI ---

def render_modification_confirmation_ui():
    """Renders the UI to confirm or discard a proposed test case modification."""
    # Check if required state exists before proceeding
    if not all(k in st.session_state for k in ['modification_app_name', 'modification_tc_id', 'proposed_modification_data', 'original_tc_data_for_diff']):
         st.error("Modification state is incomplete. Cannot render confirmation.")
         # Clear potentially partial state
         st.session_state.modification_app_name = None
         st.session_state.modification_tc_id = None
         st.session_state.proposed_modification_data = None
         st.session_state.original_tc_data_for_diff = None
         st.rerun()
         return

    prop_app = st.session_state.modification_app_name
    prop_id = st.session_state.modification_tc_id
    prop_data = st.session_state.proposed_modification_data
    orig_data = st.session_state.original_tc_data_for_diff

    st.info("Proposed modification ready for review:")
    st.write(f"**Application:** `{prop_app}` | **Test Case ID:** `{prop_id}`")

    col_o, col_p = st.columns(2)
    with col_o:
        st.write("**Original Test Case:**")
        st.json(orig_data or {"error": "Original data not found"}, expanded=False)
    with col_p:
        st.write("**Proposed Test Case:**")
        st.json(prop_data, expanded=False) # Assumes prop_data is always set here

    # Buttons for action
    discard_btn_col, apply_btn_col = st.columns([1, 1])
    with discard_btn_col:
        if st.button("❌ Discard Change", key="discard_mod_btn"):
            # Clear modification state
            st.session_state.modification_app_name = None
            st.session_state.modification_tc_id = None
            st.session_state.proposed_modification_data = None
            st.session_state.original_tc_data_for_diff = None
            st.success("Modification discarded.")
            st.rerun()

    with apply_btn_col:
        if st.button("✅ Apply Change", key="apply_mod_btn", type="primary"):
            # Ensure generated_test_cases exists and contains the app
            if 'generated_test_cases' not in st.session_state or prop_app not in st.session_state.generated_test_cases:
                 st.error(f"Cannot apply change: Results for application '{prop_app}' not found in session state.")
                 # Clear mod state as it's invalid now
                 st.session_state.modification_app_name = None
                 st.session_state.modification_tc_id = None
                 st.session_state.proposed_modification_data = None
                 st.session_state.original_tc_data_for_diff = None
                 st.rerun()
                 return # Stop processing

            app_cases = st.session_state.generated_test_cases.get(prop_app)
            applied = False
            if isinstance(app_cases, list):
                for i, tc in enumerate(app_cases):
                    # Match based on TC ID (case-sensitive comparison)
                    if isinstance(tc, dict) and tc.get("Test Case ID") == prop_id:
                        try:
                            # Directly modify the list item in session state
                            st.session_state.generated_test_cases[prop_app][i] = prop_data
                            st.success(f"Test Case '{prop_id}' in application '{prop_app}' has been updated.")
                            applied = True
                            break # Exit loop once applied
                        except IndexError:
                             st.error(f"Internal Error: Index {i} out of bounds for '{prop_app}' results.")
                             applied = False
                             break
                        except Exception as e:
                             st.error(f"Failed to apply change in session state: {e}")
                             applied = False # Ensure state isn't cleared if apply failed
                             break # Stop trying

            if not applied:
                 # Only show error if loop finished without applying, unless an error occurred above
                 if 'e' not in locals(): # Check if loop finished normally without finding ID
                    st.error(f"Could not find Test Case ID '{prop_id}' within the generated results for application '{prop_app}'.")

            # Clear modification state only if successfully applied
            if applied:
                st.session_state.modification_app_name = None
                st.session_state.modification_tc_id = None
                st.session_state.proposed_modification_data = None
                st.session_state.original_tc_data_for_diff = None
                st.rerun()


def render_modification_request_ui():
    """Renders the UI to select a test case and provide modification instructions."""
    st.subheader("✍️ Modify Existing Test Case")

    # Get apps that have successfully generated test cases (list of dicts)
    valid_apps_for_mod = [
        app for app, cases in st.session_state.get('generated_test_cases', {}).items()
        if isinstance(cases, list) and cases and all(isinstance(tc, dict) for tc in cases)
    ]

    if not valid_apps_for_mod:
        st.info("No test cases with valid data are available to modify. Please generate results first.")
        return

    # --- Selection ---
    sel_app_mod = st.selectbox(
        "Select Application:",
        options=valid_apps_for_mod,
        key="mod_app_select_widget",
        help="Choose the application containing the test case to modify."
    )

    # --- Find Original Test Case ---
    original_tc = None
    original_tc_id = None
    # Use .get with default empty list for safety
    app_test_cases = st.session_state.generated_test_cases.get(sel_app_mod, [])

    # Allow selecting TC ID via dropdown
    # Extract IDs, providing a fallback if 'Test Case ID' key is missing
    tc_ids_in_app = [
         tc.get("Test Case ID", f"MISSING_ID_INDEX_{i}")
         for i, tc in enumerate(app_test_cases) if isinstance(tc, dict)
    ]
    # Make IDs unique in case of duplicates from LLM or missing IDs
    unique_tc_ids_in_app = sorted(list(set(tc_ids_in_app)))


    if not unique_tc_ids_in_app or all(id.startswith("MISSING_ID_INDEX_") for id in unique_tc_ids_in_app):
         st.warning(f"No valid Test Case IDs found within the results for '{sel_app_mod}'. Cannot modify.")
         return # Stop rendering if no valid IDs

    selected_tc_id_mod = st.selectbox(
         "Select Test Case ID to Modify:",
         options=unique_tc_ids_in_app,
         key="mod_tcid_select_widget",
         help="Select the ID of the test case you want to change."
     )


    # Find the corresponding original data based on the selected ID
    if selected_tc_id_mod and not selected_tc_id_mod.startswith("MISSING_ID_INDEX_"):
        for i, tc in enumerate(app_test_cases):
            # Check type and ID match
            if isinstance(tc, dict) and tc.get("Test Case ID") == selected_tc_id_mod:
                original_tc = tc
                original_tc_id = selected_tc_id_mod # Store the confirmed ID
                break # Found it
        # Display original for reference if found
        if original_tc:
             st.caption("Original Test Case (for reference):")
             st.json(original_tc, expanded=False)
        else:
             # This might happen if the selected ID was somehow invalid despite dropdown
             st.error(f"Internal Error: Could not retrieve data for selected TC ID '{selected_tc_id_mod}'.")
             return # Cannot proceed without original data


    # --- Instructions ---
    mod_instructions = st.text_area(
        "Modification Instructions:",
        key="mod_instructions_widget",
        height=150,
        placeholder="e.g., 'Change the priority to High', 'Add a step to verify the confirmation email', 'Rewrite expected results for clarity'",
        help="Clearly describe the changes you want the AI to make to this test case."
    )

    # --- Action Button ---
    # Enable button only if we have valid selections and instructions
    button_disabled = not (sel_app_mod and original_tc_id and mod_instructions and original_tc)
    if st.button("🚀 Get Refactored Version", key="get_refactor_btn", disabled=button_disabled):
        # Set state for processing in the main loop
        st.session_state.refactor_request = {
            "app_name": sel_app_mod,
            "tc_id": original_tc_id,
            "instructions": mod_instructions,
            "original_data": original_tc
        }
        st.rerun() # Trigger main loop to handle the request

    # Add a hint if the button is disabled
    if button_disabled and (sel_app_mod and original_tc_id): # Check if only instructions are missing
         if not mod_instructions:
              st.caption("Please enter modification instructions.")

