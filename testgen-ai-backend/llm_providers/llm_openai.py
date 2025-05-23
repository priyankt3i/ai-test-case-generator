from typing import Dict, Tuple, Optional

# Langchain Core Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# OpenAI specific imports
try:
    from openai import AuthenticationError as OpenAIAuthenticationError, RateLimitError as OpenAIRateLimitError
except ImportError:
    # Define fallbacks if openai package is not fully available or for type hinting
    OpenAIAuthenticationError = type('OpenAIAuthenticationError', (Exception,), {})
    OpenAIRateLimitError = type('OpenAIRateLimitError', (Exception,), {})

# Import config and utilities
try:
    from config import DEFAULT_TEMPERATURE
    from helper.utils import import_class, log_message
    log_message("Successfully imported config and utils in llm_openai.", "DEBUG")
except ImportError as e:
    # This fallback log_message is only if helper.utils itself is broken
    # which should not happen if utils.py is correctly set up.
    print(f"LLM_OPENAI_PROVIDER: CRITICAL: Failed to import from config or helper.utils: {e}")
    # Define minimal fallbacks to allow type checking but operations will likely fail
    DEFAULT_TEMPERATURE = 0.7
    def log_message(msg, level, **kwargs): print(f"LLM_OPENAI_FALLBACK_LOGGER [{level}] {msg}")
    def import_class(mod, cls): return None
    log_message("Using fallback log_message/import_class in llm_openai due to import error.", "ERROR")


def _initialize_openai(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """
    Initializes OpenAI LLM and Embeddings based on provided configuration and credentials.
    """
    log_message("Initializing OpenAI provider...", "INFO")
    api_key = credentials.get("api_key")
    if not api_key:
        log_message("OpenAI init failed: API key missing.", "ERROR")
        # No st.error, just log and return. Endpoint will handle response.
        return None, None

    llm_module_name = config_dict.get("llm_module")
    llm_class_name = config_dict.get("llm_class")
    embeddings_module_name = config_dict.get("embeddings_module")
    embeddings_class_name = config_dict.get("embeddings_class")

    if not all([llm_module_name, llm_class_name, embeddings_module_name, embeddings_class_name]):
        log_message("OpenAI init failed: Module/class names missing in config_dict.", "ERROR")
        return None, None

    LLMClass = import_class(llm_module_name, llm_class_name)
    EmbeddingsClass = import_class(embeddings_module_name, embeddings_class_name)

    if not LLMClass or not EmbeddingsClass:
        log_message("OpenAI init failed: Could not import required LangChain classes. Ensure `langchain-openai` is installed.", "ERROR")
        return None, None

    try:
        llm = LLMClass(api_key=api_key, model=model_name, temperature=DEFAULT_TEMPERATURE)
        log_message(f"OpenAI LLM Class {LLMClass.__name__} initialized for model '{model_name}'.", "DEBUG")

        embeddings = EmbeddingsClass(api_key=api_key)
        log_message(f"OpenAI Embeddings Class {EmbeddingsClass.__name__} initialized.", "DEBUG")

        log_message("OpenAI provider initialized successfully.", "INFO")
        return llm, embeddings

    except OpenAIAuthenticationError as e:
        log_message(f"OpenAI Authentication Failed: {e}. Invalid API Key.", "ERROR")
        return None, None
    except OpenAIRateLimitError as e:
        log_message(f"OpenAI Rate Limit Exceeded: {e}. Check plan or wait.", "ERROR")
        return None, None
    except Exception as e:
        error_msg = f"Error initializing OpenAI: {e}"
        log_message(error_msg, "ERROR", exc_info=True)
        # Specific error logging for model not found
        if "model_not_found" in str(e).lower() or "does not exist" in str(e).lower():
            log_message(f"OpenAI Error: Model '{model_name}' not found or inaccessible with your API key.", "ERROR")
        return None, None
    finally:
        log_message("OpenAI initialization process completed.", "DEBUG")
