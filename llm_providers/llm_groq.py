# Handles initialization for the Groq provider.

import streamlit as st
from typing import Dict, Tuple, Optional

# Langchain Core Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings # Though Groq uses fallback

# Groq specific imports (if any specific exceptions are needed)
# try:
#     from groq import AuthenticationError as GroqAuthError # Example if groq sdk is used directly
# except ImportError:
#     GroqAuthError = Exception

# Import config and utilities (adjust path if needed)
try:
    from config import DEFAULT_TEMPERATURE
    # Assuming utils.py is in the same directory or accessible via PYTHONPATH
    from helper.utils import import_class, log_message
except ImportError as e:
    try: log_message(f"CRITICAL: Failed to import required modules (config, utils) in llm_groq.py: {e}", "ERROR")
    except NameError: print(f"CRITICAL: Failed to import required modules (config, utils) in llm_groq.py: {e}")
    st.error(f"CRITICAL: Failed to import required modules (config, utils) in llm_groq.py: {e}")
    # Define fallbacks or stop if critical
    DEFAULT_TEMPERATURE = 0.7
    def log_message(msg, level): print(f"{level}: {msg}")
    def import_class(mod, cls): return None
    # st.stop()


def _initialize_groq(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """
    Initializes Groq LLM based on provided configuration and credentials.
    Note: Groq typically requires fallback embeddings (e.g., OpenAI).

    Args:
        config_dict: Configuration dictionary specific to the Groq provider
                     (e.g., from LLM_PROVIDER_CONFIG).
        credentials: Dictionary containing the necessary credentials (e.g., 'api_key').
        model_name: The specific Groq model to initialize (e.g., "llama3-8b-8192").

    Returns:
        A tuple containing the initialized LLM and None (for embeddings, requires fallback),
        or (None, None) on failure.
    """
    log_message("Initializing Groq provider...", "INFO")
    api_key = credentials.get("api_key")
    if not api_key:
        log_message("Groq init failed: API key missing.", "ERROR")
        st.error("Groq Error: API key is missing.")
        return None, None

    # Dynamically import the required LangChain class for Groq
    LLMClass = import_class(config_dict.get("llm_module"), config_dict.get("llm_class"))

    if not LLMClass:
        log_message("Groq init failed: Could not import required LangChain class.", "ERROR")
        st.error("Failed to import necessary LangChain class for Groq. Ensure `langchain-groq` is installed.")
        return None, None

    try:
        # Initialize the LLM
        # Parameter names might vary slightly based on LangChain version (e.g., model vs model_name)
        # Check the LangChain Groq class documentation if needed.
        llm = LLMClass(groq_api_key=api_key, model_name=model_name, temperature=DEFAULT_TEMPERATURE)
        # Or potentially: llm = LLMClass(api_key=api_key, model=model_name, temperature=DEFAULT_TEMPERATURE)

        log_message(f"Groq LLM Class {LLMClass.__name__} initialized for model '{model_name}'.", "DEBUG")
        log_message("Groq provider initialized successfully (embeddings require fallback).", "INFO")
        # Return LLM and None for embeddings, as fallback is handled separately
        return llm, None

    except Exception as e:
        error_msg = f"Error initializing Groq: {e}"
        log_message(error_msg, "ERROR")
        # Check for common Groq error patterns (often via HTTP status codes in exceptions)
        # Note: Specific Groq exceptions might require the `groq` package itself.
        # Langchain might wrap these in more generic exceptions.
        if "authentication" in str(e).lower() or "401" in str(e):
            st.error(f"Groq Authentication Failed: Check API Key.")
        elif ("invalid_request" in str(e).lower() or "400" in str(e)) and "model" in str(e).lower():
             st.error(f"Groq Invalid Request: Model '{model_name}' might be incorrect, unavailable, or not supported.")
        elif "rate limit" in str(e).lower() or "429" in str(e):
             st.error(f"Groq Rate Limit Exceeded: Please check your Groq Cloud plan or wait.")
        elif "permission" in str(e).lower() or "403" in str(e):
             st.error(f"Groq Permission Denied: Your API key may not have access to the requested model '{model_name}'.")
        else:
            st.error(error_msg)
        return None, None
    finally:
        log_message("Groq initialization process completed.", "DEBUG")  
