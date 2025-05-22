# llm_providers/llm_embeddings_utils.py
# Contains utility functions related to embeddings, like fallback initialization.
# Adapted for backend.

from typing import Optional

# Langchain Core Imports
from langchain_core.embeddings import Embeddings

# OpenAI specific imports for fallback
try:
    from openai import AuthenticationError as OpenAIAuthenticationError
except ImportError:
    OpenAIAuthenticationError = type('OpenAIAuthenticationError', (Exception,), {}) # Fallback

# Import config and utilities
try:
    # Assuming utils.py is in the helper subfolder relative to llm_providers folder
    from helper.utils import import_class, log_message
    log_message("Successfully imported utils in llm_embeddings_utils.", "DEBUG")
except ImportError as e:
    print(f"LLM_EMBEDDINGS_UTILS: CRITICAL: Failed to import from helper.utils: {e}")
    def log_message(msg, level, **kwargs): print(f"LLM_EMBEDDINGS_UTILS_FALLBACK_LOGGER [{level}] {msg}")
    def import_class(mod, cls): return None
    log_message("Using fallback log_message/import_class in llm_embeddings_utils due to import error.", "ERROR")


def _get_fallback_embeddings(fallback_openai_key: str) -> Optional[Embeddings]:
    """
    Initializes OpenAI embeddings for fallback scenarios.
    """
    log_message("Attempting to initialize OpenAI fallback embeddings...", "INFO")
    if not fallback_openai_key:
        log_message("OpenAI fallback init failed: API key missing.", "ERROR")
        return None

    EmbeddingsClass = import_class("langchain_openai", "OpenAIEmbeddings")
    if not EmbeddingsClass:
        log_message("OpenAI fallback init failed: Could not import OpenAIEmbeddings. Ensure `langchain-openai` is installed.", "ERROR")
        return None

    try:
        embeddings = EmbeddingsClass(api_key=fallback_openai_key)
        log_message("OpenAI fallback embeddings initialized successfully.", "INFO")
        return embeddings
    except OpenAIAuthenticationError as e:
        log_message(f"OpenAI Fallback Auth Error: {e}. Invalid API Key.", "ERROR")
        return None
    except Exception as e:
        log_message(f"Error initializing OpenAI fallback embeddings: {e}", "ERROR", exc_info=True)
        return None
    # Removed duplicate RuntimeError catch, as generic Exception catch above handles it.
