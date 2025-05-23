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
    from llm_providers.llm_openrouter import _initialize_openrouter
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
        "OpenRouter": _initialize_openrouter,
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
        log_message(f"Using prompt for app identification:\\n{prompt_template_str}", "DEBUG")
        app_chain = app_prompt | llm | StrOutputParser()
        log_message("Application identification chain created.", "DEBUG")

        with st.spinner(f"Asking LLM ({provider_name} / {llm.__class__.__name__}) to identify applications..."):
            result_str = app_chain.invoke({"text": text})
            log_message("LLM invocation for identification complete.", "DEBUG")
            log_message(f"Raw LLM output for identification:\\n---\\n{result_str}\\n---", "DEBUG")

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

            if extracted_list_str is None and not result_str.strip
