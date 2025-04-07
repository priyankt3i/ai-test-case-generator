# Handles initialization for the OpenAI provider.

import streamlit as st
from typing import Dict, Tuple, Optional

# Langchain Core Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# OpenAI specific imports
try:
    from openai import AuthenticationError as OpenAIAuthenticationError, RateLimitError as OpenAIRateLimitError
except ImportError:
    OpenAIAuthenticationError, OpenAIRateLimitError = Exception, Exception # Fallback

# Import config and utilities (adjust path if needed)
try:
    from config import DEFAULT_TEMPERATURE
    # Assuming utils.py is in the same directory or accessible via PYTHONPATH
    from helper.utils import import_class, log_message
except ImportError as e:
    try: log_message(f"CRITICAL: Failed to import required modules (config, utils) in llm_openai.py: {e}", "ERROR")
    except NameError: print(f"CRITICAL: Failed to import required modules (config, utils) in llm_openai.py: {e}")
    st.error(f"CRITICAL: Failed to import required modules (config, utils) in llm_openai.py: {e}")
    # Define fallbacks or stop if critical
    DEFAULT_TEMPERATURE = 0.7
    def log_message(msg, level): print(f"{level}: {msg}")
    def import_class(mod, cls): return None
    # st.stop()


def _initialize_openai(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """
    Initializes OpenAI LLM and Embeddings based on provided configuration and credentials.

    Args:
        config_dict: Configuration dictionary specific to the OpenAI provider
                     (e.g., from LLM_PROVIDER_CONFIG).
        credentials: Dictionary containing the necessary credentials (e.g., 'api_key').
        model_name: The specific OpenAI model to initialize (e.g., "gpt-4", "gpt-3.5-turbo").

    Returns:
        A tuple containing the initialized LLM and Embeddings objects, or (None, None) on failure.
    """
    log_message("Initializing OpenAI provider...", "INFO")
    api_key = credentials.get("api_key")
    if not api_key:
        log_message("OpenAI init failed: API key missing.", "ERROR")
        st.error("OpenAI Error: API key is missing.")
        return None, None

    # Dynamically import the required LangChain classes for OpenAI
    LLMClass = import_class(config_dict.get("llm_module"), config_dict.get("llm_class"))
    EmbeddingsClass = import_class(config_dict.get("embeddings_module"), config_dict.get("embeddings_class"))

    if not LLMClass or not EmbeddingsClass:
        log_message("OpenAI init failed: Could not import required LangChain classes.", "ERROR")
        st.error("Failed to import necessary LangChain classes for OpenAI. Ensure `langchain-openai` is installed.")
        return None, None

    try:
        # Initialize the LLM
        llm = LLMClass(api_key=api_key, model=model_name, temperature=DEFAULT_TEMPERATURE)
        log_message(f"OpenAI LLM Class {LLMClass.__name__} initialized for model '{model_name}'.", "DEBUG")

        # Initialize Embeddings
        embeddings = EmbeddingsClass(api_key=api_key)
        log_message(f"OpenAI Embeddings Class {EmbeddingsClass.__name__} initialized.", "DEBUG")

        log_message("OpenAI provider initialized successfully.", "INFO")
        return llm, embeddings

    except OpenAIAuthenticationError as e:
        log_message(f"OpenAI Authentication Failed: {e}", "ERROR")
        st.error("OpenAI Authentication Failed: Invalid API Key.")
        return None, None
    except OpenAIRateLimitError as e:
        log_message(f"OpenAI Rate Limit Exceeded: {e}", "ERROR")
        st.error("OpenAI Rate Limit Exceeded. Please check your plan or wait.")
        return None, None
    except Exception as e:
        error_msg = f"Error initializing OpenAI: {e}"
        log_message(error_msg, "ERROR")
        # Check for common errors like model not found
        if "model_not_found" in str(e).lower() or "does not exist" in str(e).lower():
            st.error(f"OpenAI Error: Model '{model_name}' not found or inaccessible with your API key.")
        else:
            st.error(error_msg)
        return None, None
    finally:
        log_message("OpenAI initialization process completed.", "DEBUG")

