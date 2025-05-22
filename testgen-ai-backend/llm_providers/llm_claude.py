# llm_providers/llm_claude.py
# Handles initialization for the Anthropic Claude provider, adapted for backend.

from typing import Dict, Tuple, Optional

# Langchain Core Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings # Though Claude uses fallback

# Anthropic specific imports (if any specific exceptions are needed)
# Example: from anthropic import AuthenticationError as AnthropicAuthError
# For now, relying on generic Exception handling or errors raised by Langchain's Claude wrapper.

# Import config and utilities
try:
    from config import DEFAULT_TEMPERATURE
    from helper.utils import import_class, log_message
    log_message("Successfully imported config and utils in llm_claude.", "DEBUG")
except ImportError as e:
    print(f"LLM_CLAUDE_PROVIDER: CRITICAL: Failed to import from config or helper.utils: {e}")
    DEFAULT_TEMPERATURE = 0.7
    def log_message(msg, level, **kwargs): print(f"LLM_CLAUDE_FALLBACK_LOGGER [{level}] {msg}")
    def import_class(mod, cls): return None
    log_message("Using fallback log_message/import_class in llm_claude due to import error.", "ERROR")


def _initialize_claude(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """
    Initializes Anthropic Claude LLM. Embeddings require fallback.
    """
    log_message("Initializing Claude provider...", "INFO")
    api_key = credentials.get("api_key")
    if not api_key:
        log_message("Claude init failed: API key missing.", "ERROR")
        return None, None

    llm_module_name = config_dict.get("llm_module")
    llm_class_name = config_dict.get("llm_class")

    if not llm_module_name or not llm_class_name:
        log_message("Claude init failed: Module/class names missing in config_dict.", "ERROR")
        return None, None

    LLMClass = import_class(llm_module_name, llm_class_name)
    if not LLMClass:
        log_message("Claude init failed: Could not import LangChain class. Ensure `langchain-anthropic` is installed.", "ERROR")
        return None, None

    try:
        # Parameter name for Langchain's ChatAnthropic is 'api_key' or 'anthropic_api_key'
        # As of recent Langchain versions, it's often just 'api_key'.
        # If issues, check specific version of langchain-anthropic.
        llm = LLMClass(api_key=api_key, model_name=model_name, temperature=DEFAULT_TEMPERATURE)
        # Alternative: llm = LLMClass(anthropic_api_key=api_key, model_name=model_name, temperature=DEFAULT_TEMPERATURE)
        
        log_message(f"Claude LLM Class {LLMClass.__name__} initialized for model '{model_name}'.", "DEBUG")
        log_message("Claude provider initialized successfully (embeddings require fallback).", "INFO")
        return llm, None # Embeddings are handled by fallback mechanism

    except Exception as e:
        error_msg = f"Error initializing Claude: {e}"
        log_message(error_msg, "ERROR", exc_info=True)
        # Check for common Anthropic error patterns (these are illustrative)
        error_str_lower = str(e).lower()
        if "authentication_error" in error_str_lower or "invalid api key" in error_str_lower:
            log_message("Claude Authentication Failed: Check API Key.", "ERROR")
        elif "permission_error" in error_str_lower:
            log_message(f"Claude Permission Error: Key might lack access to model '{model_name}'.", "ERROR")
        elif "invalid_request_error" in error_str_lower and ("model" in error_str_lower or "not found" in error_str_lower):
            log_message(f"Claude Invalid Request: Model '{model_name}' might be incorrect or unavailable.", "ERROR")
        elif "rate_limit_error" in error_str_lower:
             log_message("Claude Rate Limit Error: API rate limit exceeded.", "ERROR")
        return None, None
    finally:
        log_message("Claude initialization process completed.", "DEBUG")
