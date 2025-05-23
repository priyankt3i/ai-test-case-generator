# llm_providers/llm_openrouter.py
# Handles initialization for the OpenRouter provider, adapted for backend.

from typing import Dict, Tuple, Optional

# Langchain Core Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# OpenRouter (OpenAI-compatible) specific imports
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    # Using a generic exception for API errors, as specific OpenRouter exceptions might not be exposed
    from openai import OpenAIError as OpenRouterAPIError
except ImportError:
    # Fallback for environments where langchain_openai or openai might not be installed
    ChatOpenAI = type('ChatOpenAI', (object,), {}) # Dummy class
    OpenAIEmbeddings = type('OpenAIEmbeddings', (object,), {}) # Dummy class
    OpenRouterAPIError = type('OpenRouterAPIError', (Exception,), {}) # Dummy exception
    print("LLM_OPENROUTER_PROVIDER: WARNING: 'langchain-openai' or 'openai' not found. OpenRouter functionality may be limited.")


# Import config and utilities
try:
    from config import DEFAULT_TEMPERATURE
    from helper.utils import import_class, log_message
    log_message("Successfully imported config and utils in llm_openrouter.", "DEBUG")
except ImportError as e:
    print(f"LLM_OPENROUTER_PROVIDER: CRITICAL: Failed to import from config or helper.utils: {e}")
    DEFAULT_TEMPERATURE = 0.7
    def log_message(msg, level, **kwargs): print(f"LLM_OPENROUTER_FALLBACK_LOGGER [{level}] {msg}")
    def import_class(mod, cls): return None
    log_message("Using fallback log_message/import_class in llm_openrouter due to import error.", "ERROR")


def _initialize_openrouter(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """
    Initializes OpenRouter LLM and Embeddings.
    OpenRouter uses an OpenAI-compatible API.
    """
    log_message("Initializing OpenRouter provider...", "INFO")
    api_key = credentials.get("api_key")
    if not api_key:
        log_message("OpenRouter init failed: API key missing.", "ERROR")
        return None, None

    # OpenRouter requires a base_url for the OpenAI client
    # This should be configured in your application's config_dict
    openrouter_base_url = config_dict.get("base_url", "https://openrouter.ai/api/v1")
    if not openrouter_base_url:
        log_message("OpenRouter init failed: 'base_url' missing in config_dict. It should be 'https://openrouter.ai/api/v1'.", "ERROR")
        return None, None

    # For OpenRouter, we'll directly use ChatOpenAI and OpenAIEmbeddings from langchain_openai
    # The config_dict should ideally specify these or we can hardcode them here.
    # For consistency with your original code, we'll still use import_class,
    # assuming config_dict will point to 'langchain_openai.chat_models.ChatOpenAI' etc.
    llm_module_name = config_dict.get("llm_module", "langchain_openai.chat_models")
    llm_class_name = config_dict.get("llm_class", "ChatOpenAI")
    embeddings_module_name = config_dict.get("embeddings_module", "langchain_openai.embeddings")
    embeddings_class_name = config_dict.get("embeddings_class", "OpenAIEmbeddings")

    if not all([llm_module_name, llm_class_name, embeddings_module_name, embeddings_class_name]):
        log_message("OpenRouter init failed: Module/class names missing in config_dict.", "ERROR")
        return None, None

    LLMClass = import_class(llm_module_name, llm_class_name)
    EmbeddingsClass = import_class(embeddings_module_name, embeddings_class_name)

    if not LLMClass or not EmbeddingsClass:
        log_message("OpenRouter init failed: Could not import LangChain OpenAI classes. Ensure `langchain-openai` is installed.", "ERROR")
        return None, None

    embed_model_id = config_dict.get("embeddings_model_id", "embed-lite") # A common OpenRouter embedding model

    try:
        # Initialize ChatOpenAI with OpenRouter's base_url and API key
        llm = LLMClass(
            openai_api_key=api_key,
            openai_api_base=openrouter_base_url,
            model_name=model_name, # Use model_name for ChatOpenAI
            temperature=DEFAULT_TEMPERATURE
        )
        log_message(f"OpenRouter LLM Class {LLMClass.__name__} initialized for model '{model_name}'.", "DEBUG")

        # Initialize OpenAIEmbeddings with OpenRouter's base_url and API key
        embeddings = EmbeddingsClass(
            openai_api_key=api_key,
            openai_api_base=openrouter_base_url,
            model=embed_model_id # Use model for OpenAIEmbeddings
        )
        log_message(f"OpenRouter Embeddings Class {EmbeddingsClass.__name__} initialized (model: {embed_model_id}).", "DEBUG")

        log_message("OpenRouter provider initialized successfully.", "INFO")
        return llm, embeddings

    except OpenRouterAPIError as e:
        error_detail = str(e)
        log_message(f"OpenRouter API Error: {error_detail}", "ERROR")
        if "authentication" in error_detail.lower() or "api key" in error_detail.lower():
            log_message("OpenRouter Auth Failed: API Key invalid or permissions issue. Check your OpenRouter key.", "ERROR")
        elif "quota" in error_detail.lower() or "rate limit" in error_detail.lower():
            log_message("OpenRouter API Quota/Rate Limit Exceeded.", "ERROR")
        elif "model" in error_detail.lower() and ("not found" in error_detail.lower() or "invalid" in error_detail.lower()):
            log_message(f"OpenRouter Error: Model '{model_name}' or embedding model '{embed_model_id}' not found or invalid.", "ERROR")
        return None, None
    except Exception as e:
        error_msg = f"Error initializing OpenRouter: {e}"
        log_message(error_msg, "ERROR", exc_info=True)
        return None, None
    finally:
        log_message("OpenRouter initialization process completed.", "DEBUG")

