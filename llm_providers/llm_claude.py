# Handles initialization for the Anthropic Claude provider.

import streamlit as st
from typing import Dict, Tuple, Optional

# Langchain Core Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings # Though Claude uses fallback

# Anthropic specific imports (if any specific exceptions are needed)
# try:
#     from anthropic import AuthenticationError as AnthropicAuthError # Example
# except ImportError:
#     AnthropicAuthError = Exception

# Import config and utilities (adjust path if needed)
try:
    from config import DEFAULT_TEMPERATURE
    # Assuming utils.py is in the same directory or accessible via PYTHONPATH
    from helper.utils import import_class, log_message
except ImportError as e:
    try: log_message(f"CRITICAL: Failed to import required modules (config, utils) in llm_claude.py: {e}", "ERROR")
    except NameError: print(f"CRITICAL: Failed to import required modules (config, utils) in llm_claude.py: {e}")
    st.error(f"CRITICAL: Failed to import required modules (config, utils) in llm_claude.py: {e}")
    # Define fallbacks or stop if critical
    DEFAULT_TEMPERATURE = 0.7
    def log_message(msg, level): print(f"{level}: {msg}")
    def import_class(mod, cls): return None
    # st.stop()


def _initialize_claude(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """
    Initializes Anthropic Claude LLM based on provided configuration and credentials.
    Note: Claude typically requires fallback embeddings (e.g., OpenAI).

    Args:
        config_dict: Configuration dictionary specific to the Claude provider
                     (e.g., from LLM_PROVIDER_CONFIG).
        credentials: Dictionary containing the necessary credentials (e.g., 'api_key').
        model_name: The specific Claude model to initialize (e.g., "claude-3-opus-20240229").

    Returns:
        A tuple containing the initialized LLM and None (for embeddings, requires fallback),
        or (None, None) on failure.
    """
    log_message("Initializing Claude provider...", "INFO")
    api_key = credentials.get("api_key")
    if not api_key:
        log_message("Claude init failed: API key missing.", "ERROR")
        st.error("Claude Error: API key is missing.")
        return None, None

    # Dynamically import the required LangChain class for Claude
    LLMClass = import_class(config_dict.get("llm_module"), config_dict.get("llm_class"))

    if not LLMClass:
        log_message("Claude init failed: Could not import required LangChain class.", "ERROR")
        st.error("Failed to import necessary LangChain class for Claude. Ensure `langchain-anthropic` is installed.")
        return None, None

    try:
        # Initialize the LLM
        # Parameter name might be 'anthropic_api_key' or similar depending on LangChain version
        llm = LLMClass(api_key=api_key, model=model_name, temperature=DEFAULT_TEMPERATURE)
        # Or: llm = LLMClass(anthropic_api_key=api_key, model_name=model_name, temperature=DEFAULT_TEMPERATURE)
        # Check the specific LangChain class documentation if initialization fails.

        log_message(f"Claude LLM Class {LLMClass.__name__} initialized for model '{model_name}'.", "DEBUG")
        log_message("Claude provider initialized successfully (embeddings require fallback).", "INFO")
        # Return LLM and None for embeddings, as fallback is handled separately
        return llm, None

    except Exception as e:
        error_msg = f"Error initializing Claude: {e}"
        log_message(error_msg, "ERROR")
        # Check for common Anthropic error patterns
        if "authentication_error" in str(e).lower() or "invalid api key" in str(e).lower():
            st.error(f"Claude Authentication Failed: Check API Key.")
        elif "permission_error" in str(e).lower():
            st.error(f"Claude Permission Error: Key might lack access to model '{model_name}'. Check Anthropic plan/permissions.")
        elif "invalid_request_error" in str(e).lower() and ("model" in str(e).lower() or "not found" in str(e).lower()):
            st.error(f"Claude Invalid Request: Model '{model_name}' might be incorrect, unavailable, or not supported by the API key.")
        elif "rate_limit_error" in str(e).lower():
             st.error(f"Claude Rate Limit Error: API rate limit exceeded. Please check your plan or wait.")
        else:
            st.error(error_msg)
        return None, None
    
    finally:
        log_message("Claude initialization process completed.", "DEBUG")
# --- END MODIFIED: _initialize_claude ---

