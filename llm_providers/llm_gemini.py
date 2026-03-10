# Handles initialization for the Google Gemini provider.

import streamlit as st
from typing import Dict, Tuple, Optional
import asyncio

# Langchain Core Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# Google specific imports
try:
    # Use the specific LangChain integration package name
    from google.api_core.exceptions import PermissionDenied as GooglePermissionDenied, ResourceExhausted as GoogleResourceExhausted
except ImportError:
    GooglePermissionDenied, GoogleResourceExhausted = Exception, Exception # Fallback

# Import config and utilities (adjust path if needed)
try:
    from config import DEFAULT_TEMPERATURE
    # Assuming utils.py is in the same directory or accessible via PYTHONPATH
    from helper.utils import import_class, log_message
except ImportError as e:
    try: log_message(f"CRITICAL: Failed to import required modules (config, utils) in llm_gemini.py: {e}", "ERROR")
    except NameError: print(f"CRITICAL: Failed to import required modules (config, utils) in llm_gemini.py: {e}")
    st.error(f"CRITICAL: Failed to import required modules (config, utils) in llm_gemini.py: {e}")
    # Define fallbacks or stop if critical
    DEFAULT_TEMPERATURE = 0.7
    def log_message(msg, level): print(f"{level}: {msg}")
    def import_class(mod, cls): return None
    # st.stop()


def _initialize_gemini(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """
    Initializes Google Gemini LLM and Embeddings based on provided configuration and credentials.

    Args:
        config_dict: Configuration dictionary specific to the Gemini provider
                     (e.g., from LLM_PROVIDER_CONFIG).
        credentials: Dictionary containing the necessary credentials (e.g., 'api_key').
        model_name: The specific Gemini model to initialize (e.g., "gemini-pro").

    Returns:
        A tuple containing the initialized LLM and Embeddings objects, or (None, None) on failure.
    """
    log_message("Initializing Gemini provider...", "INFO")
    
    # Ensure an asyncio event loop is running in the current thread
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # 'RuntimeError: There is no current event loop...'
        log_message("No asyncio event loop found, creating a new one for the current thread.", "INFO")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    api_key = credentials.get("api_key")
    if not api_key:
        log_message("Gemini init failed: API key missing.", "ERROR")
        st.error("Gemini Error: API key is missing.")
        return None, None

    # Dynamically import the required LangChain classes for Gemini
    LLMClass = import_class(config_dict.get("llm_module"), config_dict.get("llm_class"))
    EmbeddingsClass = import_class(config_dict.get("embeddings_module"), config_dict.get("embeddings_class"))

    if not LLMClass or not EmbeddingsClass:
        log_message("Gemini init failed: Could not import required LangChain classes.", "ERROR")
        st.error("Failed to import necessary LangChain classes for Gemini. Ensure `langchain-google-genai` is installed.")
        return None, None

    selected_embedding_model = None
    try:
        # Initialize the LLM
        # Note: `convert_system_message_to_human=True` might be needed depending on LangChain version and Gemini model behavior
        llm = LLMClass(google_api_key=api_key, model=model_name, temperature=DEFAULT_TEMPERATURE, convert_system_message_to_human=True)
        log_message(f"Gemini LLM Class {LLMClass.__name__} initialized for model '{model_name}'.", "DEBUG")

        # Initialize Embeddings with candidate fallback probing.
        configured_model = config_dict.get("embeddings_model_id")
        configured_candidates = config_dict.get("embeddings_model_ids", [])
        default_candidates = [
            "models/gemini-embedding-001",
            "gemini-embedding-001",
            "models/text-embedding-004",
            "text-embedding-004",
            "models/embedding-001",
            "embedding-001",
        ]

        candidates = []
        if configured_model:
            candidates.append(configured_model)
        if isinstance(configured_candidates, list):
            candidates.extend(configured_candidates)
        candidates.extend(default_candidates)

        # De-duplicate while preserving order.
        dedup_candidates = []
        for candidate in candidates:
            if candidate and candidate not in dedup_candidates:
                dedup_candidates.append(candidate)

        embeddings = None
        candidate_errors = []
        for candidate_model in dedup_candidates:
            try:
                probe_embeddings = EmbeddingsClass(model=candidate_model, google_api_key=api_key)
                # Probe once so unsupported models fail fast here, not during generation.
                probe_vector = probe_embeddings.embed_query("ping")
                if not probe_vector:
                    raise ValueError("Embedding probe returned empty vector.")
                embeddings = probe_embeddings
                selected_embedding_model = candidate_model
                log_message(
                    f"Gemini Embeddings Class {EmbeddingsClass.__name__} initialized and probed (model: {candidate_model}).",
                    "DEBUG"
                )
                break
            except Exception as candidate_error:
                candidate_errors.append(f"{candidate_model}: {candidate_error}")

        if embeddings is None:
            error_msg = (
                "Failed to initialize Gemini embeddings for candidate models "
                f"{dedup_candidates}. Last error: {candidate_errors[-1] if candidate_errors else 'Unknown error'}"
            )
            log_message(f"Gemini Embeddings init failed: {error_msg}", "ERROR")
            # Return LLM so non-RAG operations (like identify) can still run.
            st.error(f"Gemini Embeddings Error: {error_msg}")
            return llm, None

        log_message("Gemini provider initialized successfully.", "INFO")
        return llm, embeddings

    except (GooglePermissionDenied, ValueError) as e:
        # Handle permission errors (often related to API key validity or enablement) and value errors
        log_message(f"Gemini Permission/Value Error: {e}", "ERROR")
        if "api key not valid" in str(e).lower():
            st.error("Gemini Authentication Failed: API Key is not valid. Please check your key and ensure the Google AI API is enabled in your cloud console.")
        elif "permission denied" in str(e).lower():
             st.error(f"Gemini Permission Denied: Check if the API key has access to model '{model_name}' and embedding model '{selected_embedding_model or 'auto-detected candidate'}'. Error: {e}")
        else:
            st.error(f"Gemini Permission/Value Error: {e}")
        return None, None
    except GoogleResourceExhausted as e:
        # Handle quota errors
        log_message(f"Gemini API Quota Exceeded: {e}", "ERROR")
        st.error("Gemini API Quota Exceeded. Please check your usage limits in the Google Cloud Console.")
        return None, None
    except Exception as e:
        # Catch any other initialization errors
        error_msg = f"Error initializing Gemini: {e}"
        log_message(error_msg, "ERROR")
        # Check for common errors like model not found
        if "model" in str(e).lower() and ("not found" in str(e).lower() or "permission denied" in str(e).lower()):
             st.error(
                 f"Gemini Error: Model '{model_name}' or embedding model "
                 f"'{selected_embedding_model or 'auto-detected candidate'}' not found/inaccessible with your API key."
             )
        else:
            st.error(error_msg)
        return None, None
    finally:
        log_message("Gemini initialization process completed.", "DEBUG")
# --- END MODIFIED: _initialize_gemini ---
