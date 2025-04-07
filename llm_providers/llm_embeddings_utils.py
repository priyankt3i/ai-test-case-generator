# Contains utility functions related to embeddings, like fallback initialization.

import streamlit as st
from typing import Optional

# Langchain Core Imports
from langchain_core.embeddings import Embeddings

# Specific exception imports (adjust path if utils.py is elsewhere)
try:
    # Assuming utils.py is in the same directory or accessible via PYTHONPATH
    from helper.utils import import_class, log_message
except ImportError as e:
    # Basic fallback logging if utils isn't available during this split
    print(f"ERROR: Failed to import from utils: {e}")
    def log_message(msg, level): print(f"{level}: {msg}")
    def import_class(mod, cls): return None # Or raise error
    st.error(f"CRITICAL: Failed to import required modules (utils): {e}")
    # Consider stopping if utils is absolutely critical here
    # st.stop()

# OpenAI specific imports for fallback
try:
    from openai import AuthenticationError as OpenAIAuthenticationError
except ImportError:
    OpenAIAuthenticationError = Exception # Fallback if openai package not installed

def _get_fallback_embeddings(fallback_openai_key: str) -> Optional[Embeddings]:
    """
    Initializes OpenAI embeddings for fallback scenarios (e.g., for providers
    like Claude or Groq that don't have their own native LangChain embeddings).

    Args:
        fallback_openai_key: The OpenAI API key to use for fallback embeddings.

    Returns:
        An initialized OpenAIEmbeddings instance or None if initialization fails.
    """
    log_message("Attempting to initialize OpenAI fallback embeddings...", "INFO")
    if not fallback_openai_key:
        log_message("OpenAI fallback init failed: API key missing.", "ERROR")
        st.error("RAG requires an OpenAI API key for fallback embeddings, but it's missing.")
        return None

    # Dynamically import the OpenAIEmbeddings class
    EmbeddingsClass = import_class("langchain_openai", "OpenAIEmbeddings")
    if not EmbeddingsClass:
        log_message("OpenAI fallback init failed: Could not import OpenAIEmbeddings.", "ERROR")
        st.error("Failed to import OpenAIEmbeddings for fallback. Ensure `langchain-openai` is installed.")
        return None

    try:
        # Initialize the embeddings class
        embeddings = EmbeddingsClass(api_key=fallback_openai_key)
        log_message("OpenAI fallback embeddings initialized successfully.", "INFO")
        return embeddings
    except OpenAIAuthenticationError as e:
        log_message(f"OpenAI Fallback Auth Error: {e}", "ERROR")
        st.error("OpenAI Fallback Auth Error: Invalid API Key provided for fallback embeddings.")
        return None
    except Exception as e:
        # Catch any other initialization errors
        log_message(f"Error initializing OpenAI fallback embeddings: {e}", "ERROR")
        st.error(f"Error initializing OpenAI fallback embeddings: {e}")
        return None

    except RuntimeError as e:
        log_message(f"Runtime Error initializing OpenAI fallback embeddings: {e}", "ERROR")
        st.error(f"Runtime Error initializing OpenAI fallback embeddings: {e}")
        return None 