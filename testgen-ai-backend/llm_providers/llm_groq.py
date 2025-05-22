# llm_providers/llm_groq.py
# Handles initialization for the Groq provider, adapted for backend.

from typing import Dict, Tuple, Optional

# Langchain Core Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings # Though Groq uses fallback

# Groq specific imports (if any specific exceptions are needed)
# Example: from groq import AuthenticationError as GroqAuthError
# For now, relying on generic Exception handling or errors raised by Langchain's Groq wrapper.

# Import config and utilities
try:
    from config import DEFAULT_TEMPERATURE
    from helper.utils import import_class, log_message
    log_message("Successfully imported config and utils in llm_groq.", "DEBUG")
except ImportError as e:
    print(f"LLM_GROQ_PROVIDER: CRITICAL: Failed to import from config or helper.utils: {e}")
    DEFAULT_TEMPERATURE = 0.7
    def log_message(msg, level, **kwargs): print(f"LLM_GROQ_FALLBACK_LOGGER [{level}] {msg}")
    def import_class(mod, cls): return None
    log_message("Using fallback log_message/import_class in llm_groq due to import error.", "ERROR")


def _initialize_groq(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """
    Initializes Groq LLM. Embeddings require fallback.
    """
    log_message("Initializing Groq provider...", "INFO")
    api_key = credentials.get("api_key")
    if not api_key:
        log_message("Groq init failed: API key missing.", "ERROR")
        return None, None

    llm_module_name = config_dict.get("llm_module")
    llm_class_name = config_dict.get("llm_class")

    if not llm_module_name or not llm_class_name:
        log_message("Groq init failed: Module/class names missing in config_dict.", "ERROR")
        return None, None

    LLMClass = import_class(llm_module_name, llm_class_name)
    if not LLMClass:
        log_message("Groq init failed: Could not import LangChain class. Ensure `langchain-groq` is installed.", "ERROR")
        return None, None

    try:
        # LangChain's ChatGroq class typically takes 'groq_api_key'
        llm = LLMClass(groq_api_key=api_key, model_name=model_name, temperature=DEFAULT_TEMPERATURE)
        
        log_message(f"Groq LLM Class {LLMClass.__name__} initialized for model '{model_name}'.", "DEBUG")
        log_message("Groq provider initialized successfully (embeddings require fallback).", "INFO")
        return llm, None # Embeddings are handled by fallback mechanism

    except Exception as e:
        error_msg = f"Error initializing Groq: {e}"
        log_message(error_msg, "ERROR", exc_info=True)
        error_str_lower = str(e).lower()
        if "authentication" in error_str_lower or "401" in error_str_lower:
            log_message("Groq Authentication Failed: Check API Key.", "ERROR")
        elif ("invalid_request" in error_str_lower or "400" in error_str_lower) and "model" in error_str_lower:
             log_message(f"Groq Invalid Request: Model '{model_name}' might be incorrect or unavailable.", "ERROR")
        elif "rate limit" in error_str_lower or "429" in error_str_lower:
             log_message("Groq Rate Limit Exceeded.", "ERROR")
        elif "permission" in error_str_lower or "403" in error_str_lower:
             log_message(f"Groq Permission Denied for model '{model_name}'.", "ERROR")
        return None, None
    finally:
        log_message("Groq initialization process completed.", "DEBUG")
