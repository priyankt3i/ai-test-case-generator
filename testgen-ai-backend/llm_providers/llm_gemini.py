# llm_providers/llm_gemini.py
# Handles initialization for the Google Gemini provider, adapted for backend.

from typing import Dict, Tuple, Optional

# Langchain Core Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# Google specific imports
try:
    from google.api_core.exceptions import PermissionDenied as GooglePermissionDenied, ResourceExhausted as GoogleResourceExhausted
except ImportError:
    GooglePermissionDenied = type('GooglePermissionDenied', (Exception,), {})
    GoogleResourceExhausted = type('GoogleResourceExhausted', (Exception,), {})

# Import config and utilities
try:
    from config import DEFAULT_TEMPERATURE
    from helper.utils import import_class, log_message
    log_message("Successfully imported config and utils in llm_gemini.", "DEBUG")
except ImportError as e:
    print(f"LLM_GEMINI_PROVIDER: CRITICAL: Failed to import from config or helper.utils: {e}")
    DEFAULT_TEMPERATURE = 0.7
    def log_message(msg, level, **kwargs): print(f"LLM_GEMINI_FALLBACK_LOGGER [{level}] {msg}")
    def import_class(mod, cls): return None
    log_message("Using fallback log_message/import_class in llm_gemini due to import error.", "ERROR")


def _initialize_gemini(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """
    Initializes Google Gemini LLM and Embeddings.
    """
    log_message("Initializing Gemini provider...", "INFO")
    api_key = credentials.get("api_key")
    if not api_key:
        log_message("Gemini init failed: API key missing.", "ERROR")
        return None, None

    llm_module_name = config_dict.get("llm_module")
    llm_class_name = config_dict.get("llm_class")
    embeddings_module_name = config_dict.get("embeddings_module")
    embeddings_class_name = config_dict.get("embeddings_class")

    if not all([llm_module_name, llm_class_name, embeddings_module_name, embeddings_class_name]):
        log_message("Gemini init failed: Module/class names missing in config_dict.", "ERROR")
        return None, None

    LLMClass = import_class(llm_module_name, llm_class_name)
    EmbeddingsClass = import_class(embeddings_module_name, embeddings_class_name)

    if not LLMClass or not EmbeddingsClass:
        log_message("Gemini init failed: Could not import LangChain classes. Ensure `langchain-google-genai` is installed.", "ERROR")
        return None, None

    embed_model_id = config_dict.get("embeddings_model_id", "models/embedding-001") # Define here for use in error messages

    try:
        llm = LLMClass(google_api_key=api_key, model=model_name, temperature=DEFAULT_TEMPERATURE, convert_system_message_to_human=True)
        log_message(f"Gemini LLM Class {LLMClass.__name__} initialized for model '{model_name}'.", "DEBUG")

        embeddings = EmbeddingsClass(model=embed_model_id, google_api_key=api_key)
        log_message(f"Gemini Embeddings Class {EmbeddingsClass.__name__} initialized (model: {embed_model_id}).", "DEBUG")

        log_message("Gemini provider initialized successfully.", "INFO")
        return llm, embeddings

    except (GooglePermissionDenied, ValueError) as e:
        error_detail = str(e)
        log_message(f"Gemini Permission/Value Error: {error_detail}", "ERROR")
        if "api key not valid" in error_detail.lower():
            log_message("Gemini Auth Failed: API Key invalid. Check key and ensure Google AI API is enabled.", "ERROR")
        elif "permission denied" in error_detail.lower():
             log_message(f"Gemini Permission Denied: Check API key access for model '{model_name}' or embedding model '{embed_model_id}'. Error: {error_detail}", "ERROR")
        # else: log_message already logged the generic error
        return None, None
    except GoogleResourceExhausted as e:
        log_message(f"Gemini API Quota Exceeded: {e}", "ERROR")
        return None, None
    except Exception as e:
        error_msg = f"Error initializing Gemini: {e}"
        log_message(error_msg, "ERROR", exc_info=True)
        if "model" in str(e).lower() and ("not found" in str(e).lower() or "permission denied" in str(e).lower()):
             log_message(f"Gemini Error: Model '{model_name}' or embedding model '{embed_model_id}' not found/inaccessible.", "ERROR")
        return None, None
    finally:
        log_message("Gemini initialization process completed.", "DEBUG")
