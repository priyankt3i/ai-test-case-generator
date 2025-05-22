# llm_providers/llm_ollama.py
# Handles initialization for the Ollama provider, adapted for backend.

import requests # For checking server reachability
from typing import Dict, Tuple, Optional

# Langchain Core Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# Ollama specific imports
try:
    from langchain_ollama import ChatOllama
    from langchain_ollama import OllamaEmbeddings
    LANGCHAIN_OLLAMA_AVAILABLE = True
except ImportError:
    LANGCHAIN_OLLAMA_AVAILABLE = False
    class ChatOllama: pass # Dummy for type hinting if import fails
    class OllamaEmbeddings: pass # Dummy

# Import config and utilities
try:
    from config import DEFAULT_TEMPERATURE
    from helper.utils import log_message # No import_class needed here as we import directly
    log_message("Successfully imported config and utils in llm_ollama.", "DEBUG")
except ImportError as e:
    print(f"LLM_OLLAMA_PROVIDER: CRITICAL: Failed to import from config or helper.utils: {e}")
    DEFAULT_TEMPERATURE = 0.7
    def log_message(msg, level, **kwargs): print(f"LLM_OLLAMA_FALLBACK_LOGGER [{level}] {msg}")
    log_message("Using fallback log_message in llm_ollama due to import error.", "ERROR")


def _initialize_ollama(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """
    Initializes Ollama LLM and Embeddings.
    """
    log_message("Initializing Ollama provider...", "INFO")

    if not LANGCHAIN_OLLAMA_AVAILABLE:
        log_message("Ollama init failed: langchain-ollama package not available. `pip install langchain-ollama`", "ERROR")
        return None, None

    base_url = credentials.get("base_url", "http://localhost:11434").strip()
    if not base_url:
        base_url = "http://localhost:11434" # Default if empty after strip
        log_message("Ollama Base URL empty, using default: http://localhost:11434", "WARNING")

    if not model_name:
        log_message("Ollama init failed: Model name missing.", "ERROR")
        return None, None
    log_message(f"Ollama using Base URL: {base_url}, Model: {model_name}", "DEBUG")

    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        log_message(f"Successfully connected to Ollama server at {base_url}", "DEBUG")
        # server_models = response.json().get('models', [])
        # if not any(m['name'].startswith(model_name) for m in server_models):
        #     log_message(f"Model '{model_name}' not found on Ollama server {base_url}. Ensure it's pulled.", "WARNING")
    except requests.exceptions.ConnectionError:
        log_message(f"Ollama Connection Error: Could not connect to {base_url}. Is Ollama running?", "ERROR")
        return None, None
    except requests.exceptions.Timeout:
        log_message(f"Ollama Connection Timeout: Server at {base_url} unresponsive.", "ERROR")
        return None, None
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 'N/A'
        log_message(f"Ollama Request Error: Failed to query {base_url}. Status: {status_code}. Error: {e}", "ERROR", exc_info=True)
        return None, None

    try:
        llm = ChatOllama(
            base_url=base_url,
            model=model_name,
            temperature=DEFAULT_TEMPERATURE
        )
        log_message("Ollama LLM (ChatOllama) initialized.", "DEBUG")

        embeddings = OllamaEmbeddings(
            base_url=base_url,
            model=model_name # Typically use same model for embeddings with Ollama
        )
        log_message(f"Ollama Embeddings (OllamaEmbeddings) initialized (model: {model_name}).", "DEBUG")

        try:
            log_message(f"Attempting test invocation of Ollama model '{model_name}'...", "DEBUG")
            llm.invoke("Respond with just 'OK'")
            log_message(f"Ollama model '{model_name}' responded successfully.", "DEBUG")
        except Exception as model_e:
            error_detail = str(model_e).lower()
            if "model not found" in error_detail or "library not found" in error_detail:
                 error_msg = f"Ollama Model Error: Model '{model_name}' not found at {base_url}. Pull it with `ollama pull {model_name}`. Error: {model_e}"
            elif "connection refused" in error_detail:
                 error_msg = f"Ollama Connection Error during test: Could not connect to {base_url}."
            else:
                 error_msg = f"Ollama Model Error: Test invocation failed for model '{model_name}' at {base_url}. Check Ollama server logs. Error: {model_e}"
            log_message(error_msg, "ERROR", exc_info=True)
            return None, None

        log_message("Ollama provider initialized successfully.", "INFO")
        return llm, embeddings

    except Exception as e:
        log_message(f"Error initializing Ollama LangChain components: {e}", "ERROR", exc_info=True)
        return None, None
    finally:
        log_message("Ollama initialization process completed.", "DEBUG")
