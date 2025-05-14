# llm_integration_core.py
# Main module for LangChain setup, LLM interactions (identification, RAG, refactoring).
# Imports provider-specific initialization logic from the 'llm_providers' subfolder.

import streamlit as st
import yaml
import os
import json
import re # Import regex module
import ast # For literal_eval fallback in parsing
from typing import Dict, Any, Tuple, List, Optional
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Langchain Core Imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# Import config and utilities (adjust path if needed)
try:
    # Assuming config.py and utils.py are in the parent directory
    # relative to where llm_integration_core.py is run, or in PYTHONPATH
    import config
    from config import (
        LLM_PROVIDER_CONFIG, FALLBACK_EMBEDDING_PROVIDERS,
        APP_CONTEXT_FOLDER_PATH, NO_CONTEXT_OPTION, # NO_CONTEXT_OPTION might not be needed here anymore
        CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_SEARCH_K,
        EXCEL_EXPECTED_COLUMNS
    )
    # Assuming utils.py is in the helper subfolder or accessible via PYTHONPATH
    from helper.utils import parse_json_output, log_message
except ImportError as e:
    print(f"CRITICAL: Failed to import required modules (config, helper.utils) in core: {e}")
    # Define dummy log_message if utils fails
    def log_message(msg, level, **kwargs): print(f"[{level}] {msg}") # Basic logger without exc_info
    # Define a basic parse_json_output if utils fails, to avoid NameError
    def parse_json_output(s, expected_type=None):
        log_message(f"Utils import failed, using basic parse_json_output for: {s[:50]}...", "WARNING")
        try:
            data = json.loads(s)
            if expected_type and not isinstance(data, expected_type): return None
            return data
        except Exception:
            try:
                data = ast.literal_eval(s)
                if expected_type and not isinstance(data, expected_type): return None
                return data
            except Exception:
                return None
    st.error(f"CRITICAL: Failed to import required modules (config, helper.utils): {e}")
    st.stop()

# --- Provider-Specific Initialization Imports ---
try:
    from llm_providers.llm_openai import _initialize_openai
    from llm_providers.llm_gemini import _initialize_gemini
    from llm_providers.llm_claude import _initialize_claude
    from llm_providers.llm_bedrock import _initialize_bedrock
    from llm_providers.llm_groq import _initialize_groq
    from llm_providers.llm_ollama import _initialize_ollama
    from llm_providers.llm_embeddings_utils import _get_fallback_embeddings
    log_message("Successfully imported provider initialization functions from 'llm_providers' folder.", "DEBUG")
except ImportError as e:
    log_message(f"CRITICAL: Failed to import one or more provider initialization modules from 'llm_providers': {e}", "ERROR")
    st.error(f"CRITICAL: Failed to import provider logic from 'llm_providers'. Ensure the folder exists and contains the correct files (e.g., llm_openai.py): {e}")
    # Define dummy functions to prevent NameErrors later
    def _initialize_openai(*args): return None, None
    def _initialize_gemini(*args): return None, None
    def _initialize_claude(*args): return None, None
    def _initialize_bedrock(*args): return None, None
    def _initialize_groq(*args): return None, None
    def _initialize_ollama(*args): return None, None
    def _get_fallback_embeddings(*args): return None
    st.stop() # Stop execution if imports fail


# --- Main Initialization Function ---
def get_llm_and_embeddings(provider: str, model_name: str, credentials: Dict, fallback_openai_key: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """
    Initializes LangChain LLM and Embeddings objects based on the selected provider.
    Delegates the actual initialization to provider-specific functions imported
    from the 'llm_providers' subfolder.
    """
    log_message(f"Getting LLM and Embeddings for provider: {provider}, model: {model_name}", "INFO")
    llm: Optional[BaseChatModel] = None
    embeddings: Optional[Embeddings] = None

    config_dict = LLM_PROVIDER_CONFIG.get(provider)
    if not config_dict:
        log_message(f"Invalid provider selected: {provider}. No configuration found.", "ERROR")
        st.error(f"Invalid provider selected: '{provider}'. Configuration missing.")
        return None, None

    init_functions = {
        "OpenAI": _initialize_openai,
        "Gemini": _initialize_gemini,
        "Claude": _initialize_claude,
        "AWS Bedrock": _initialize_bedrock,
        "Groq": _initialize_groq,
        "Ollama": _initialize_ollama,
    }
    init_func = init_functions.get(provider)

    if not init_func:
        log_message(f"Initialization function not found for provider: {provider}", "ERROR")
        st.error(f"Internal Error: Initialization logic not defined for provider: {provider}")
        return None, None

    log_message(f"Calling initialization function for {provider}...", "DEBUG")
    try:
        llm, embeddings = init_func(config_dict, credentials, model_name)
    except Exception as e:
        # *** REMOVED exc_info=True ***
        log_message(f"Unexpected error calling initialization function for {provider}: {e}", "ERROR")
        st.error(f"An unexpected error occurred while trying to initialize {provider}: {e}")
        return None, None

    is_fallback_provider = provider in FALLBACK_EMBEDDING_PROVIDERS
    log_message(f"Provider: {provider}, Is fallback needed? {is_fallback_provider}, LLM Initialized: {llm is not None}, Embeddings Initialized: {embeddings is not None}", "DEBUG")

    if is_fallback_provider and llm and not embeddings:
        log_message(f"Provider {provider} requires fallback embeddings. Attempting initialization.", "INFO")
        embeddings = _get_fallback_embeddings(fallback_openai_key)
        if not embeddings:
            log_message(f"Failed to initialize fallback embeddings (OpenAI) for {provider}.", "WARNING")
            st.warning(f"RAG may fail for {provider}. Fallback embeddings (using OpenAI) could not be initialized. Check the fallback OpenAI API key in settings.")
            return llm, None
        else:
             log_message(f"Successfully initialized fallback embeddings for {provider}.", "INFO")

    if not llm:
        log_message(f"LLM initialization final check failed for {provider}. Returning None, None.", "ERROR")
        return None, None

    if not embeddings:
        if not is_fallback_provider:
            log_message(f"Embeddings initialization final check failed for non-fallback provider {provider}. RAG will not work.", "ERROR")
            st.error(f"Embeddings initialization failed for {provider}. RAG features will not work. Check logs for details.")
            return llm, None

    log_message(f"LLM and Embeddings initialized successfully for {provider}.", "INFO")
    return llm, embeddings


# --- Credential Checking ---
def check_credentials(provider: str, credentials: Dict, fallback_key: str, require_fallback_for_rag: bool) -> Tuple[bool, str]:
    """
    Validates if all necessary credentials are present for the selected provider
    based on the configuration in LLM_PROVIDER_CONFIG.
    """
    log_message(f"Checking credentials for provider: {provider}", "DEBUG")
    config_dict = LLM_PROVIDER_CONFIG.get(provider)
    if not config_dict:
        return False, f"Invalid provider selected: '{provider}'. No configuration found."

    missing = []
    required_creds = config_dict.get("credentials", [])

    for key in required_creds:
        if provider == "Ollama" and key in ["base_url", "model"]:
            continue
        if provider == "AWS Bedrock" and key == "aws_session_token":
            key_id = credentials.get("aws_access_key_id", "")
            token = credentials.get("aws_session_token", "")
            if key_id.startswith("ASIA") and not token:
                missing.append("AWS Session Token (Required for temporary credentials starting with 'ASIA')")
            continue
        if not credentials.get(key, "").strip():
            friendly_key_name = key.replace("_", " ").title()
            missing.append(friendly_key_name)

    needs_fallback = provider in FALLBACK_EMBEDDING_PROVIDERS
    if require_fallback_for_rag and needs_fallback and not fallback_key.strip():
        missing.append("OpenAI API Key (Required for RAG Fallback Embeddings with this provider)")

    if missing:
        error_msg = f"Missing credentials/settings for {provider}: {', '.join(missing)}."
        log_message(error_msg, "WARNING")
        return False, error_msg

    log_message(f"Credential check passed for {provider}.", "DEBUG")
    return True, ""


# --- Prompt Template Fetching ---
def get_prompt_template(provider_name: str, template_key: str) -> str:
    """
    Gets the appropriate prompt template string for the given provider and task key.
    Falls back to the default template defined directly in the config module
    if no provider-specific override exists in LLM_PROVIDER_CONFIG.
    """
    if 'config' not in globals():
        raise ImportError("Configuration module 'config' not loaded.")

    provider_config = config.LLM_PROVIDER_CONFIG.get(provider_name, {})
    provider_prompts = provider_config.get("prompt_templates", {})
    if template_key in provider_prompts:
        log_message(f"Using '{template_key}' prompt override for provider '{provider_name}'", "DEBUG")
        return provider_prompts[template_key]

    default_template_name = f"{template_key}_PROMPT_TEMPLATE"
    if hasattr(config, default_template_name):
        log_message(f"Using default '{template_key}' prompt (from config.{default_template_name}) for provider '{provider_name}'", "DEBUG")
        return getattr(config, default_template_name)
    else:
        error_msg = f"Prompt template key '{template_key}' requested, but no override found for provider '{provider_name}' AND the default template variable 'config.{default_template_name}' is missing."
        log_message(error_msg, "ERROR")
        # Fallback to a generic error message if template is missing
        generic_template = f"Error: Prompt template '{template_key}' not found for provider '{provider_name}' or in defaults."
        log_message(f"Returning generic error template: {generic_template}", "ERROR")
        return generic_template
        # raise AttributeError(error_msg) # Avoid raising exception, return error string instead


# --- Helper: Extract list-like content from LLM output ---
def _extract_list_from_llm_output(raw_output: str) -> Optional[str]:
    """
    Attempts to extract a Python list string ('[...]') or reconstruct one
    from bullet points or markdown code blocks in potentially messy LLM output.
    """
    log_message("Attempting to extract list structure from raw LLM output...", "DEBUG")
    if not raw_output or not raw_output.strip():
        log_message("Extraction skipped: Raw output is empty.", "DEBUG")
        return None

    code_block_patterns = [
        r"```(?:python|json)\s*(\[.*?\])\s*```",
        r"```\s*(\[.*?\])\s*```"
    ]
    extracted_from_block = None
    for pattern in code_block_patterns:
        match = re.search(pattern, raw_output, re.DOTALL | re.IGNORECASE)
        if match:
            extracted_from_block = match.group(1).strip()
            log_message(f"Found potential list within markdown code block (pattern: {pattern}): '{extracted_from_block[:100]}...'", "DEBUG")
            try:
                # Try parsing as JSON first
                json.loads(extracted_from_block)
                log_message("List extracted from code block confirmed via JSON parsing.", "DEBUG")
                return extracted_from_block
            except json.JSONDecodeError:
                try:
                    # Fallback to ast.literal_eval for Python list literals
                    ast.literal_eval(extracted_from_block)
                    log_message("List extracted from code block confirmed via ast.literal_eval.", "DEBUG")
                    return extracted_from_block
                except (ValueError, SyntaxError) as e:
                    log_message(f"Content within code block is not valid JSON or Python list literal: {e}. Discarding this match.", "WARNING")
                    extracted_from_block = None
                    continue
        if extracted_from_block: break

    if extracted_from_block: return extracted_from_block

    log_message("Markdown code block extraction failed or content was invalid.", "DEBUG")

    log_message("Trying to find direct list literal (not necessarily in code block)...", "DEBUG")
    direct_match = re.search(r'(\[.*?\])', raw_output, re.DOTALL)
    if direct_match:
        extracted_direct = direct_match.group(1).strip()
        log_message(f"Found potential direct list literal via regex search: '{extracted_direct[:100]}...'", "DEBUG")
        try:
            json.loads(extracted_direct)
            log_message("Direct list literal confirmed via JSON parsing.", "DEBUG")
            return extracted_direct
        except json.JSONDecodeError:
            try:
                ast.literal_eval(extracted_direct)
                log_message("Direct list literal confirmed via ast.literal_eval.", "DEBUG")
                return extracted_direct
            except (ValueError, SyntaxError) as e:
                log_message(f"Direct list literal found by search is not valid JSON or Python list literal: {e}. Trying bullet points.", "WARNING")

    log_message("Direct list literal search failed or invalid. Trying bullet point extraction.", "DEBUG")
    bullet_items = []
    bullet_pattern = re.compile(r'^\s*[\*\-]\s+(.*)', re.MULTILINE)
    matches = bullet_pattern.findall(raw_output)

    if matches:
        log_message(f"Found {len(matches)} potential bullet point items.", "DEBUG")
        for item in matches:
            cleaned_item = item.strip().strip("'\"`")
            escaped_item = json.dumps(cleaned_item) # Escape for JSON list construction
            if cleaned_item:
                bullet_items.append(escaped_item)

        if bullet_items:
            list_string = f"[{', '.join(bullet_items)}]"
            log_message(f"Constructed list from bullets: {list_string}", "DEBUG")
            try:
                # Validate the constructed JSON string
                json.loads(list_string)
                log_message("Constructed list from bullets confirmed via JSON parsing.", "DEBUG")
                return list_string
            except json.JSONDecodeError as e:
                 log_message(f"Constructed list from bullets failed JSON validation (unexpected): {e}", "ERROR")
        else:
            log_message("Found bullet markers but no valid content extracted after cleaning.", "DEBUG")

    log_message("Could not extract a valid list structure using any method (code block, direct, bullets).", "WARNING")
    return None


# --- LLM Interaction Functions ---

def identify_applications(text: str, llm: BaseChatModel, provider_name: str) -> List[str]:
    """
    Identifies application names from text using the provided LLM and provider-specific prompt.
    Includes improved pre-processing to extract list data from potentially messy output before parsing.
    """
    log_message(f"Starting application identification using provider: {provider_name}...", "INFO")
    if not text or not text.strip():
        log_message("Identification failed: Input text is empty or whitespace.", "ERROR")
        st.error("Cannot identify applications: Input text is empty.")
        return []
    if not llm:
        log_message("Identification failed: LLM is not initialized.", "ERROR")
        st.error("Cannot identify applications: LLM is not initialized.")
        return []

    try:
        template_key = "IDENTIFY_APP"
        prompt_template_str = get_prompt_template(provider_name, template_key)
        if prompt_template_str.startswith("Error:"): # Check if template fetching failed
             st.error(prompt_template_str)
             return []
        app_prompt = ChatPromptTemplate.from_template(prompt_template_str)
        log_message(f"Using prompt for app identification:\n{prompt_template_str}", "DEBUG")
        app_chain = app_prompt | llm | StrOutputParser()
        log_message("Application identification chain created.", "DEBUG")

        with st.spinner(f"Asking LLM ({provider_name} / {llm.__class__.__name__}) to identify applications..."):
            result_str = app_chain.invoke({"text": text})
            log_message("LLM invocation for identification complete.", "DEBUG")
            log_message(f"Raw LLM output for identification:\n---\n{result_str}\n---", "DEBUG")

        if not result_str or not result_str.strip():
            log_message("LLM returned empty response for identification.", "WARNING")
            st.warning("LLM returned an empty response. Could not identify applications.")
            return []

        extracted_list_str = _extract_list_from_llm_output(result_str)

        parsed_apps = None
        if extracted_list_str:
            log_message("Using extracted list string for JSON/AST parsing.", "INFO")
            parsed_apps = parse_json_output(extracted_list_str, expected_type=list)
        else:
            log_message("List extraction failed. Attempting to parse raw LLM output directly (might fail).", "WARNING")
            parsed_apps = parse_json_output(result_str, expected_type=list)

        if parsed_apps is None:
            log_message("Failed to parse application list from LLM response (after extraction attempt).", "WARNING")
            st.warning("LLM response for applications was not in the expected list format (e.g., ['App1', 'App2']).")

            if extracted_list_str is None and not result_str.strip().startswith(("[", "{")) and ',' in result_str:
                possible_apps = [app.strip().strip("'\"`") for app in result_str.split(',') if app.strip()]
                if possible_apps:
                    log_message(f"Attempting basic comma parsing as final fallback. Found: {possible_apps}", "INFO")
                    st.info("LLM output wasn't structured, attempting basic comma parsing.")
                    final_apps = sorted(list(set(app for app in possible_apps if app)))
                    log_message(f"Identification successful via fallback comma parsing. Found apps: {final_apps}", "INFO")
                    return final_apps
                else:
                    log_message("Comma fallback attempted but yielded no apps.", "WARNING")
            else:
                 log_message("Skipping comma fallback as primary parsing failed or string looked like JSON/list.", "DEBUG")

            st.error("Could not reliably identify applications from the LLM response format. Please check the raw response in the logs.")
            return []

        cleaned_apps = [str(app).strip() for app in parsed_apps if isinstance(app, (str, int, float)) and str(app).strip()]

        if not cleaned_apps:
            log_message("Parsed list was empty or contained only non-string/empty items.", "WARNING")
            st.warning("LLM identified an empty list of applications.")
            return []

        final_apps = sorted(list(set(cleaned_apps)))
        log_message(f"Identification successful. Found apps: {final_apps}", "INFO")
        return final_apps

    except Exception as e:
        # *** REMOVED exc_info=True ***
        log_message(f"Exception during application identification: {type(e).__name__} - {e}", "ERROR")
        st.error(f"An error occurred during application identification LLM call: {e}")
        return []


# --- generate_test_cases Function ---
def generate_test_cases(
    text: str,
    selected_apps: List[str],
    uploaded_context_files: Dict[str, List[UploadedFile]],
    llm: BaseChatModel,
    embeddings: Embeddings,
    provider_name: str
) -> Dict[str, Any]:
    """
    Generates test cases using RAG, incorporating text extracted from uploaded context files.
    """
    log_message(f"--- Entered generate_test_cases using provider: {provider_name} ---", "DEBUG")
    results = {}

    # --- Initial checks ---
    if not selected_apps:
        log_message("Generate skipped: No applications selected.", "WARNING")
        return results
    if not text or not text.strip():
        log_message("Generate failed: Source text is empty.", "ERROR")
        return {app: "Error: Source text is empty." for app in selected_apps}
    if not llm:
        log_message("Generate failed: LLM is not initialized.", "ERROR")
        return {app: "Error: LLM not initialized." for app in selected_apps}
    if not embeddings:
        log_message("Generate failed: Embeddings are not initialized (required for RAG).", "ERROR")
        return {app: "Error: Embeddings not initialized (required for RAG)." for app in selected_apps}

    # Define FILE_PROCESSING_AVAILABLE based on the availability of file processing functionality
    try:
        from helper.file_processing import extract_text_from_file
        FILE_PROCESSING_AVAILABLE = True
    except ImportError:
        FILE_PROCESSING_AVAILABLE = False

    # *** MOVED CHECK: Check for file processing availability AFTER the try-except block ***
    if not FILE_PROCESSING_AVAILABLE:
        log_message("File processing functions not available. Cannot process uploaded context.", "ERROR")
        st.error("Internal Error: File processing functions are missing.")
        return {app: "Error: File processing module unavailable." for app in selected_apps}


    # --- Vector Store Creation ---
    vectorstore = None
    try:
        log_message("Creating vector store from text...", "DEBUG")
        from langchain.text_splitter import RecursiveCharacterTextSplitter # Example import
        from langchain_community.vectorstores import FAISS # Example import
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) # Use config values
        splits = text_splitter.split_text(text)
        if not splits: raise ValueError("Text splitting resulted in zero chunks.")
        vectorstore = FAISS.from_texts(texts=splits, embedding=embeddings)
        if vectorstore is None: raise ValueError("Vector store creation failed (remained None).")
        log_message("--- Vector Store Creation Succeeded ---", "INFO")
    except Exception as e:
        # *** REMOVED exc_info=True ***
        log_message(f"--- Exception during Vector Store Creation: {type(e).__name__} - {e} ---", "ERROR")
        st.error(f"Error creating vector store/embeddings: {e}.")
        return {app: f"Error creating embeddings/vector store: {e}" for app in selected_apps}


    # --- RAG Chain Setup ---
    retrieval_chain = None
    try:
        # Placeholder: Replace with your actual RAG chain setup logic
        # Ensure this part is correctly implemented in your version
        from langchain.chains.combine_documents import create_stuff_documents_chain # Example import
        from langchain.chains import create_retrieval_chain # Example import
        from langchain_core.prompts import ChatPromptTemplate # Example import
        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_SEARCH_K}) # Use config value
        template_key = "GENERATE_TC"
        prompt_template_str = get_prompt_template(provider_name, template_key)
        if prompt_template_str.startswith("Error:"): # Check if template fetching failed
             raise ValueError(prompt_template_str)
        # Format prompt (ensure EXCEL_EXPECTED_COLUMNS is available)
        if not isinstance(EXCEL_EXPECTED_COLUMNS, list) or not EXCEL_EXPECTED_COLUMNS:
             raise ValueError("EXCEL_EXPECTED_COLUMNS not defined correctly in config.")
        tc_fields = ", ".join([f"`{col}`" for col in EXCEL_EXPECTED_COLUMNS])
        formatted_prompt_str = prompt_template_str.format(field_names=tc_fields, context="{context}", input="{input}") 
        test_case_prompt = ChatPromptTemplate.from_template(formatted_prompt_str)
        document_chain = create_stuff_documents_chain(llm, test_case_prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        # End Placeholder
        if retrieval_chain is None: raise ValueError("RAG chain setup failed (remained None).")
        log_message("create_retrieval_chain Succeeded. RAG setup complete.", "INFO")
    except Exception as e:
        # *** REMOVED exc_info=True ***
        log_message(f"--- Exception during RAG Chain Setup: {type(e).__name__} - {e} ---", "ERROR")
        st.error(f"Fatal error setting up RAG chain components: {e}")
        error_message_for_results = f"Error setting up RAG chain: {e}"
        for app in selected_apps:
             if app not in results: results[app] = error_message_for_results
        return results


    # --- Generation Loop ---
    total_selected_apps = len(selected_apps)
    log_message(f"Starting generation loop for {total_selected_apps} selected apps.", "INFO")
    progress_bar = st.progress(0.0, text="Initializing test case generation...")

    for i, app_name in enumerate(selected_apps):
        log_message(f"Processing app {i+1}/{total_selected_apps}: {app_name}", "INFO")
        progress_value = (i + 1) / total_selected_apps
        progress_text = f"Generating for '{app_name}' ({i+1}/{total_selected_apps})..."
        progress_bar.progress(progress_value, text=progress_text)

        with st.status(f"Processing '{app_name}'...", expanded=False) as status:
            try:
                # *** NEW CONTEXT PROCESSING LOGIC ***
                additional_context_str = ""
                files_for_app = uploaded_context_files.get(app_name, [])
                log_message(f"App '{app_name}': Found {len(files_for_app)} uploaded context files.", "DEBUG")
                status.write(f"Processing {len(files_for_app)} uploaded context files...")

                if files_for_app:
                    for uploaded_file in files_for_app:
                        if uploaded_file is None or not hasattr(uploaded_file, 'name'):
                            log_message(f"App '{app_name}': Skipping invalid file object in list.", "WARNING")
                            continue

                        try:
                            status.write(f"- Reading `{uploaded_file.name}`...")
                            log_message(f"App '{app_name}': Extracting context from '{uploaded_file.name}' (type: {getattr(uploaded_file, 'type', 'N/A')}, size: {getattr(uploaded_file, 'size', 'N/A')})", "DEBUG")
                            extracted_text = extract_text_from_file(uploaded_file) # Assumes imported correctly

                            if extracted_text:
                                additional_context_str += f"\n\n--- Context from {uploaded_file.name} ---\n{extracted_text}\n--- End Context ---\n"
                                log_message(f"App '{app_name}': Successfully extracted and appended context from '{uploaded_file.name}'.", "DEBUG")
                            else:
                                log_message(f"App '{app_name}': No text content extracted from '{uploaded_file.name}'.", "WARNING")
                                status.warning(f"No text content extracted from `{uploaded_file.name}`.")

                        except Exception as extract_err:
                            # *** REMOVED exc_info=True ***
                            log_message(f"App '{app_name}': Failed to process context file '{uploaded_file.name}': {extract_err}", "ERROR")
                            status.warning(f"⚠️ Error reading context file `{uploaded_file.name}`: {extract_err}")

                else:
                     status.write("No context files uploaded for this app.")
                # *** END NEW CONTEXT PROCESSING LOGIC ***

                # --- Prepare Input and Invoke Chain ---
                input_query_string = f"Generate detailed test cases specifically for the application or system named: '{app_name}'. " \
                                     f"Use the retrieved requirements context below and the additional context provided (if any) to inform the test cases. " \
                                     f"Focus on requirements relevant to '{app_name}'." \
                                     f"{additional_context_str}"

                log_message(f"App '{app_name}': Prepared input query for RAG chain.", "DEBUG")

                status.write("Invoking RAG chain with LLM...")
                log_message(f"App '{app_name}': Invoking retrieval_chain...", "DEBUG")

                # Ensure retrieval_chain is not None before invoking
                if retrieval_chain is None:
                     raise RuntimeError("RAG chain was not initialized correctly.")
                response = retrieval_chain.invoke({"input": input_query_string})

                log_message(f"App '{app_name}': Received response from retrieval_chain.", "DEBUG")
                status.write("Received response from LLM.")

                # --- Process Response ---
                # *** REMOVED local try-except for parse_json_output import ***
                # Assumes parse_json_output is available from top-level import or fallback

                if isinstance(response, dict) and "answer" in response and response["answer"] and isinstance(response["answer"], str):
                    answer_str = response["answer"].strip()
                    log_message(f"App '{app_name}': Extracted 'answer' string. Length: {len(answer_str)}", "DEBUG")

                    parsed_cases = parse_json_output(answer_str, expected_type=list)

                    if parsed_cases is not None and isinstance(parsed_cases, list):
                         valid_cases = [item for item in parsed_cases if isinstance(item, dict)]
                         log_message(f"App '{app_name}': Parsed {len(parsed_cases)} items, filtered to {len(valid_cases)} valid dicts.", "DEBUG")
                         if valid_cases:
                              results[app_name] = valid_cases
                              log_message(f"App '{app_name}': Successfully extracted {len(valid_cases)} valid test cases.", "INFO")
                              status.update(label=f"✓ Generated {len(valid_cases)} cases for '{app_name}'", state="complete", expanded=False)
                         else:
                              results[app_name] = "Warning: LLM response parsed as a list, but contained no valid test case objects (dictionaries)."
                              log_message(f"App '{app_name}': Parsed list contained no valid dictionary items.", "WARNING")
                              status.update(label=f"⚠️ No Valid Cases Found for '{app_name}'", state="warning", expanded=True)
                    else:
                         results[app_name] = "Error: Failed to parse JSON list of test cases from LLM response."
                         log_message(f"App '{app_name}': Failed to parse JSON list from answer string.", "ERROR")
                         status.update(label=f"⚠️ JSON Parse Error for '{app_name}'", state="error", expanded=True)
                else:
                    results[app_name] = "Error: LLM provided no answer or the response structure was unexpected."
                    log_message(f"App '{app_name}': No 'answer' key in response or response structure invalid. Response type: {type(response)}", "ERROR")
                    status.update(label=f"⚠️ No Answer/Bad Format for '{app_name}'", state="error", expanded=True)

            except Exception as e:
                # *** REMOVED exc_info=True ***
                log_message(f"--- Exception during Generation Loop for '{app_name}': {type(e).__name__} - {e} ---", "ERROR")
                st.error(f"An error occurred during generation for '{app_name}': {e}")
                results[app_name] = f"Error: Generation failed - {e}"
                status.update(label=f"❌ Failed generation for '{app_name}'", state="error", expanded=True)

    # --- Final Steps ---
    progress_bar.empty()
    log_message("--- Finished generate_test_cases ---", "DEBUG")
    return results

# --- refactor_single_test_case Function ---
def refactor_single_test_case(
    app_name: str,
    tc_id: str,
    instructions: str,
    original_tc_data: Dict,
    llm: BaseChatModel,
    provider_name: str
) -> Optional[Dict]:
    """
    Uses the LLM to refactor a single test case based on user instructions
    and a provider-specific prompt. Expects the LLM to return a JSON object
    representing the updated test case.
    """
    log_message(f"Starting refactor for TC '{tc_id}' in app '{app_name}' using provider {provider_name}...", "INFO")

    if not llm:
        st.error("Cannot refactor: LLM is not initialized.")
        log_message("Refactor failed: LLM not initialized.", "ERROR")
        return None
    if not original_tc_data or not isinstance(original_tc_data, dict):
        st.error(f"Cannot refactor: Invalid original test case data provided for TC ID '{tc_id}'. Expected a dictionary.")
        log_message(f"Refactor failed: Invalid original data for TC '{tc_id}'. Type: {type(original_tc_data)}", "ERROR")
        return None
    if not instructions or not instructions.strip():
        st.info(f"Refactor skipped for TC '{tc_id}': Modification instructions are empty.")
        log_message(f"Refactor skipped: Empty instructions provided for TC '{tc_id}'.", "INFO")
        return None

    try:
        try:
            original_json_str = json.dumps(original_tc_data, separators=(',', ':'))
        except TypeError as te:
            log_message(f"Refactor failed for TC '{tc_id}': Could not serialize original data to JSON: {te}", "ERROR")
            st.error(f"Error preparing original test case '{tc_id}' for refactoring: Could not convert data to JSON. {te}")
            return None

        template_key = "REFACTOR_TC"
        prompt_template_str = get_prompt_template(provider_name, template_key)
        if prompt_template_str.startswith("Error:"): # Check if template fetching failed
             st.error(prompt_template_str)
             return None
        prompt = ChatPromptTemplate.from_template(prompt_template_str)
        log_message(f"Using prompt for refactoring TC '{tc_id}':\n{prompt_template_str}", "DEBUG")
        chain = prompt | llm | StrOutputParser()
        log_message(f"Refactor chain created for TC '{tc_id}'.", "DEBUG")

        with st.spinner(f"Asking LLM ({provider_name}) to refactor Test Case '{tc_id}'..."):
            log_message(f"Invoking refactor chain for TC '{tc_id}'...", "DEBUG")
            response_str = chain.invoke({
                "tc_id": tc_id,
                "original_tc_json": original_json_str,
                "user_instructions": instructions
            })
            log_message(f"Refactor chain invocation complete for TC '{tc_id}'.", "DEBUG")
            log_message(f"Raw LLM output for refactor:\n---\n{response_str}\n---", "DEBUG")

        if not response_str or not response_str.strip():
            log_message(f"LLM returned empty response for refactoring TC '{tc_id}'.", "WARNING")
            st.warning(f"LLM returned an empty response for refactoring TC '{tc_id}'. No changes applied.")
            return None

        log_message(f"Attempting to parse refactor response as JSON object: '{response_str[:200]}...'", "DEBUG")
        updated_tc_data = parse_json_output(response_str, expected_type=dict)

        if updated_tc_data is None:
            log_message(f"Refactor failed for TC '{tc_id}': Failed to parse JSON object response.", "ERROR")
            st.error(f"Refactoring failed for TC '{tc_id}': LLM response was not a valid JSON object representing the test case.")
            return None

        if not isinstance(updated_tc_data, dict):
             log_message(f"Refactor failed for TC '{tc_id}': Parsed result is not a dictionary (Type: {type(updated_tc_data)}).", "ERROR")
             st.error(f"Refactoring failed for TC '{tc_id}': Unexpected result format after parsing.")
             return None

        new_tc_id = updated_tc_data.get("Test Case ID")
        original_id = original_tc_data.get("Test Case ID", tc_id)

        if new_tc_id is not None and str(new_tc_id) != str(original_id):
            log_message(f"LLM changed TC ID during refactor from '{original_id}' to '{new_tc_id}'. Reverting.", "WARNING")
            st.warning(f"LLM attempted to change the Test Case ID from '{original_id}' to '{new_tc_id}'. The original ID has been restored.")
            updated_tc_data["Test Case ID"] = original_id
        elif "Test Case ID" not in updated_tc_data:
             log_message(f"LLM removed TC ID during refactor for '{original_id}'. Re-adding.", "WARNING")
             st.warning(f"LLM removed the Test Case ID during refactoring. The original ID '{original_id}' has been re-added.")
             updated_tc_data["Test Case ID"] = original_id

        log_message(f"Refactor successful for TC '{tc_id}'.", "INFO")
        st.success(f"Test Case '{tc_id}' refactored successfully based on instructions.")
        return updated_tc_data

    except Exception as e:
        # *** REMOVED exc_info=True ***
        log_message(f"Exception during refactoring process for TC '{tc_id}': {type(e).__name__} - {e}", "ERROR")
        st.error(f"An unexpected error occurred during the refactoring process for TC '{tc_id}': {e}")
        return None

# --- NEW: refactor_all_test_cases Function ---
def refactor_all_test_cases(
    app_name: str,
    instructions: str,
    original_tc_list: List[Dict],
    llm: BaseChatModel,
    provider_name: str
) -> Optional[List[Dict]]:
    """
    Uses the LLM to refactor ALL test cases in a list based on user instructions.
    Expects the LLM to return a JSON list of updated test case dictionaries.
    """
    log_message(f"Starting bulk refactor for app '{app_name}' using provider {provider_name}...", "INFO")

    if not llm:
        st.error("Cannot refactor: LLM is not initialized.")
        log_message("Bulk refactor failed: LLM not initialized.", "ERROR")
        return None
    if not original_tc_list or not isinstance(original_tc_list, list):
        st.error(f"Cannot refactor: Invalid original test case list provided for app '{app_name}'. Expected a list of dictionaries.")
        log_message(f"Bulk refactor failed: Invalid original data for app '{app_name}'. Type: {type(original_tc_list)}", "ERROR")
        return None
    if not all(isinstance(tc, dict) for tc in original_tc_list):
        st.error(f"Cannot refactor: Original test case list for app '{app_name}' contains non-dictionary items.")
        log_message(f"Bulk refactor failed: Original list for app '{app_name}' contains non-dict items.", "ERROR")
        return None
    if not instructions or not instructions.strip():
        st.info(f"Bulk refactor skipped for app '{app_name}': Modification instructions are empty.")
        log_message(f"Bulk refactor skipped: Empty instructions provided for app '{app_name}'.", "INFO")
        return None # Or return original_tc_list? Returning None indicates no attempt made.

    try:
        try:
            # Serialize the entire list of original test cases
            original_list_json_str = json.dumps(original_tc_list, separators=(',', ':'))
        except TypeError as te:
            log_message(f"Bulk refactor failed for app '{app_name}': Could not serialize original list to JSON: {te}", "ERROR")
            st.error(f"Error preparing original test cases for app '{app_name}' for refactoring: Could not convert list to JSON. {te}")
            return None

        # Use a new template key for bulk refactoring
        template_key = "REFACTOR_ALL_TC"
        prompt_template_str = get_prompt_template(provider_name, template_key)
        if prompt_template_str.startswith("Error:"): # Check if template fetching failed
             st.error(prompt_template_str)
             return None
        prompt = ChatPromptTemplate.from_template(prompt_template_str)
        log_message(f"Using prompt for bulk refactoring app '{app_name}':\n{prompt_template_str}", "DEBUG")
        chain = prompt | llm | StrOutputParser()
        log_message(f"Bulk refactor chain created for app '{app_name}'.", "DEBUG")

        with st.spinner(f"Asking LLM ({provider_name}) to refactor all test cases for '{app_name}'..."):
            log_message(f"Invoking bulk refactor chain for app '{app_name}'...", "DEBUG")
            response_str = chain.invoke({
                "app_name": app_name,
                "original_tc_list_json": original_list_json_str,
                "user_instructions": instructions
            })
            log_message(f"Bulk refactor chain invocation complete for app '{app_name}'.", "DEBUG")
            log_message(f"Raw LLM output for bulk refactor:\n---\n{response_str}\n---", "DEBUG")

        if not response_str or not response_str.strip():
            log_message(f"LLM returned empty response for bulk refactoring app '{app_name}'.", "WARNING")
            st.warning(f"LLM returned an empty response for bulk refactoring app '{app_name}'. No changes applied.")
            return None

        log_message(f"Attempting to parse bulk refactor response as JSON list: '{response_str[:200]}...'", "DEBUG")
        # Use the helper to extract list structure first
        extracted_list_str = _extract_list_from_llm_output(response_str)
        updated_tc_list = None
        if extracted_list_str:
             log_message("Using extracted list string for JSON/AST parsing.", "INFO")
             updated_tc_list = parse_json_output(extracted_list_str, expected_type=list)
        else:
             log_message("List extraction failed. Attempting to parse raw LLM output directly (might fail).", "WARNING")
             updated_tc_list = parse_json_output(response_str, expected_type=list)


        if updated_tc_list is None:
            log_message(f"Bulk refactor failed for app '{app_name}': Failed to parse JSON list response.", "ERROR")
            st.error(f"Bulk refactoring failed for app '{app_name}': LLM response was not a valid JSON list of test cases.")
            return None

        if not isinstance(updated_tc_list, list):
             log_message(f"Bulk refactor failed for app '{app_name}': Parsed result is not a list (Type: {type(updated_tc_list)}).", "ERROR")
             st.error(f"Bulk refactoring failed for app '{app_name}': Unexpected result format after parsing.")
             return None

        # Basic validation: Check if the result is a list of dictionaries
        valid_refactored_cases = [tc for tc in updated_tc_list if isinstance(tc, dict)]
        if len(valid_refactored_cases) != len(updated_tc_list):
             log_message(f"Bulk refactor for app '{app_name}': Result list contained non-dictionary items. Filtering them out.", "WARNING")
             st.warning("Some items returned by the LLM were not valid test case dictionaries and have been removed.")

        # Optional: More robust validation (e.g., check if IDs match original list)
        # For now, just return the list of dictionaries found.
        log_message(f"Bulk refactor successful for app '{app_name}'. Returned {len(valid_refactored_cases)} test cases.", "INFO")
        st.success(f"All test cases for '{app_name}' refactored successfully based on instructions.")
        return valid_refactored_cases

    except Exception as e:
        # *** REMOVED exc_info=True ***
        log_message(f"Exception during bulk refactoring process for app '{app_name}': {type(e).__name__} - {e}", "ERROR")
        st.error(f"An unexpected error occurred during the bulk refactoring process for app '{app_name}': {e}")
        return None

# --- NEW: perform_ai_test_case_review Function ---
def perform_ai_test_case_review(
    main_requirements_text: str,
    additional_context_str: str,
    existing_test_cases: List[Dict],
    llm: BaseChatModel,
    provider_name: str
) -> Optional[Dict]:
    """
    Uses the LLM to review a list of existing test cases against requirements and context.
    The LLM is expected to return a JSON object detailing coverage, new suggestions,
    modifications, and duplicates.
    """
    log_message(f"Starting AI test case review using provider: {provider_name}...", "INFO")

    if not llm:
        st.error("Cannot perform AI review: LLM is not initialized.")
        log_message("AI review failed: LLM not initialized.", "ERROR")
        return None
    if not main_requirements_text or not main_requirements_text.strip():
        st.error("Cannot perform AI review: Main requirements text is empty.")
        log_message("AI review failed: Main requirements text is empty.", "ERROR")
        return None
    # additional_context_str can be empty, so no check for that.
    if not isinstance(existing_test_cases, list): # Could be empty list, that's fine
        st.error("Cannot perform AI review: Existing test cases data is not a list.")
        log_message(f"AI review failed: existing_test_cases is not a list. Type: {type(existing_test_cases)}", "ERROR")
        return None

    try:
        # Prepare inputs for the prompt
        try:
            existing_test_cases_json_str = json.dumps(existing_test_cases, separators=(',', ':'))
        except TypeError as te:
            log_message(f"AI review failed: Could not serialize existing test cases to JSON: {te}", "ERROR")
            st.error(f"Error preparing existing test cases for AI review: Could not convert data to JSON. {te}")
            return None

        if not isinstance(EXCEL_EXPECTED_COLUMNS, list) or not EXCEL_EXPECTED_COLUMNS:
            log_message("AI review failed: EXCEL_EXPECTED_COLUMNS not defined correctly in config.", "ERROR")
            st.error("Configuration Error: EXCEL_EXPECTED_COLUMNS is missing or invalid.")
            return None
        field_names_str = ", ".join([f"`{col}`" for col in EXCEL_EXPECTED_COLUMNS])

        template_key = "AI_REVIEW_TC" # Matches the new template in config.py
        prompt_template_str = get_prompt_template(provider_name, template_key)
        if prompt_template_str.startswith("Error:"): # Check if template fetching failed
             st.error(prompt_template_str)
             log_message(f"AI review failed: Prompt template '{template_key}' could not be loaded.", "ERROR")
             return None

        # Format the prompt template string
        # The new prompt has placeholders: {field_names}, {{main_requirements}}, {{additional_context}}, {{existing_test_cases_json}}
        # We need to ensure these are correctly substituted.
        # Langchain's ChatPromptTemplate.from_template handles {{...}} style placeholders.
        # We manually insert field_names as it's part of the fixed structure description within the prompt.
        formatted_prompt_str = prompt_template_str.format(field_names=field_names_str)
        prompt = ChatPromptTemplate.from_template(formatted_prompt_str)

        log_message(f"Using prompt for AI test case review:\n{formatted_prompt_str[:500]}...", "DEBUG") # Log beginning of prompt
        chain = prompt | llm | StrOutputParser()
        log_message("AI review chain created.", "DEBUG")

        with st.spinner(f"Asking LLM ({provider_name}) to review test cases..."):
            log_message("Invoking AI review chain...", "DEBUG")
            response_str = chain.invoke({
                "main_requirements": main_requirements_text,
                "additional_context": additional_context_str,
                "existing_test_cases_json": existing_test_cases_json_str
            })
            log_message("AI review chain invocation complete.", "DEBUG")
            log_message(f"Raw LLM output for AI review:\n---\n{response_str}\n---", "DEBUG")

        if not response_str or not response_str.strip():
            log_message("LLM returned empty response for AI review.", "WARNING")
            st.warning("LLM returned an empty response for the AI review. No analysis available.")
            return None

        log_message(f"Attempting to parse AI review response as JSON object: '{response_str[:200]}...'", "DEBUG")
        # The AI_REVIEW_TC_PROMPT_TEMPLATE asks for a single JSON object.
        # _extract_list_from_llm_output is for lists, so we use parse_json_output directly.
        review_results_dict = parse_json_output(response_str, expected_type=dict)

        if review_results_dict is None:
            log_message("AI review failed: Failed to parse JSON object response from LLM.", "ERROR")
            st.error("AI review failed: LLM response was not a valid JSON object as expected.")
            return None

        if not isinstance(review_results_dict, dict):
             log_message(f"AI review failed: Parsed result is not a dictionary (Type: {type(review_results_dict)}).", "ERROR")
             st.error("AI review failed: Unexpected result format after parsing.")
             return None

        # Basic validation of expected top-level keys (can be expanded)
        expected_top_keys = ["coverage_summary", "newly_suggested_test_cases", "modified_test_cases_suggestions", "identified_duplicates"]
        for key in expected_top_keys:
            if key not in review_results_dict:
                log_message(f"AI review warning: Expected key '{key}' not found in LLM response.", "WARNING")
                st.warning(f"AI review result is missing the expected section: '{key}'. The output might be incomplete.")
                # Initialize missing keys as empty lists or strings to prevent downstream errors
                if key.endswith("_summary"):
                    review_results_dict[key] = "Summary not provided by AI."
                else:
                    review_results_dict[key] = []


        log_message("AI review successful. Parsed LLM response.", "INFO")
        st.success("AI Test Case Review complete.")
        return review_results_dict

    except Exception as e:
        log_message(f"Exception during AI test case review process: {type(e).__name__} - {e}", "ERROR")
        st.error(f"An unexpected error occurred during the AI test case review process: {e}")
        return None
