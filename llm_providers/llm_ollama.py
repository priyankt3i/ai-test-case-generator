# Handles initialization for the Ollama provider.

import streamlit as st
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
    # Define dummy classes if import fails, so the rest of the file can load
    # although initialization will fail later if these are actually used.
    class ChatOllama: pass
    class OllamaEmbeddings: pass


# Import config and utilities (adjust path if needed)
try:
    from config import DEFAULT_TEMPERATURE
    # Assuming utils.py is in the same directory or accessible via PYTHONPATH
    from helper.utils import log_message # No import_class needed here as we import directly
except ImportError as e:
    try: log_message(f"CRITICAL: Failed to import required modules (config, utils) in llm_ollama.py: {e}", "ERROR")
    except NameError: print(f"CRITICAL: Failed to import required modules (config, utils) in llm_ollama.py: {e}")
    st.error(f"CRITICAL: Failed to import required modules (config, utils) in llm_ollama.py: {e}")
    # Define fallbacks or stop if critical
    DEFAULT_TEMPERATURE = 0.7
    def log_message(msg, level): print(f"{level}: {msg}")
    # st.stop()


def get_ollama_models(base_url: str) -> list[str]:
    """
    Fetches the list of available models from an Ollama server.

    Args:
        base_url: The base URL of the Ollama server.

    Returns:
        A list of model names.
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model["name"] for model in models]
    except requests.exceptions.RequestException as e:
        log_message(f"Could not fetch Ollama models from {base_url}: {e}", "ERROR")
        return []

def _initialize_ollama(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """
    Initializes Ollama LLM and Embeddings based on provided configuration and credentials.

    Args:
        config_dict: Configuration dictionary specific to the Ollama provider (currently unused but kept for consistency).
        credentials: Dictionary containing Ollama settings (e.g., 'base_url').
        model_name: The specific Ollama model to use (e.g., "llama3", "mistral").

    Returns:
        A tuple containing the initialized LLM and Embeddings objects, or (None, None) on failure.
    """
    log_message("Initializing Ollama provider...", "INFO")

    # Check if the necessary package was imported successfully
    if not LANGCHAIN_OLLAMA_AVAILABLE:
        log_message("Ollama init failed: langchain-ollama package not available.", "ERROR")
        st.error("Ollama integration requires the `langchain-ollama` package. Please install it (`pip install langchain-ollama`).")
        return None, None

    # Get base URL, providing a default if not specified or empty
    base_url = credentials.get("base_url", "http://localhost:11434").strip()
    if not base_url:
        base_url = "http://localhost:11434"
        log_message("Ollama Base URL empty, using default: http://localhost:11434", "WARNING")

    # Model name is crucial for Ollama
    if not model_name:
        log_message("Ollama init failed: Model name missing.", "ERROR")
        st.error("Ollama requires a model to be selected (e.g., 'llama3', 'mistral').")
        return None, None
    log_message(f"Ollama using Base URL: {base_url}, Model: {model_name}", "DEBUG")

    # --- Check Ollama Server Reachability ---
    try:
        # Use a timeout to prevent hanging indefinitely
        response = requests.get(f"{base_url}/api/tags", timeout=5) # Query available models endpoint
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        log_message(f"Successfully connected to Ollama server at {base_url}", "DEBUG")
        # Optionally check if the specific model exists in the response tags
        # server_models = response.json().get('models', [])
        # if not any(m['name'].startswith(model_name) for m in server_models):
        #     log_message(f"Model '{model_name}' not found on Ollama server {base_url}. Available models might differ.", "WARNING")
        #     st.warning(f"Model '{model_name}' not listed on Ollama server {base_url}. Ensure it's pulled (`ollama pull {model_name}`).")

    except requests.exceptions.ConnectionError:
        error_msg = f"Ollama Connection Error: Could not connect to the Ollama server at {base_url}. Is the Ollama service running and accessible?"
        log_message(error_msg, "ERROR")
        st.error(error_msg)
        return None, None
    except requests.exceptions.Timeout:
        error_msg = f"Ollama Connection Timeout: The server at {base_url} did not respond within 5 seconds. It might be slow or unresponsive."
        log_message(error_msg, "ERROR")
        st.error(error_msg)
        return None, None
    except requests.exceptions.RequestException as e: # Catch other request errors (like HTTPError)
        status_code = e.response.status_code if e.response is not None else 'N/A'
        error_msg = f"Ollama Request Error: Failed to query the Ollama server at {base_url}. Status Code: {status_code}. Error: {e}"
        log_message(error_msg, "ERROR")
        st.error(error_msg)
        return None, None

    # --- Initialize LangChain Components ---
    try:
        # Initialize ChatOllama LLM
        llm = ChatOllama(
            base_url=base_url,
            model=model_name,
            temperature=DEFAULT_TEMPERATURE
            # Add other parameters like num_ctx, top_k, top_p if needed from config
        )
        log_message(f"Ollama LLM Class ChatOllama initialized.", "DEBUG")

        # Initialize Ollama Embeddings
        # Typically use the same model for embeddings unless specified otherwise
        # Could add an 'embeddings_model' key to credentials/config if needed
        embeddings_model = model_name
        embeddings = OllamaEmbeddings(
            base_url=base_url,
            model=embeddings_model
        )
        log_message(f"Ollama Embeddings Class OllamaEmbeddings initialized (using model: {embeddings_model}).", "DEBUG")

        # --- Test Model Invocation (Optional but Recommended) ---
        # This helps catch model-specific errors early (e.g., model not pulled)
        try:
            # Use a simple, non-empty prompt for testing
            log_message(f"Attempting test invocation of Ollama model '{model_name}'...", "DEBUG")
            llm.invoke("Respond with just 'OK'") # Keep the test prompt very simple
            log_message(f"Ollama model '{model_name}' responded successfully to test invocation.", "DEBUG")
        except Exception as model_e:
            # Improve error message for common issues
            error_detail = str(model_e).lower()
            if "model not found" in error_detail or "library not found" in error_detail:
                 error_msg = f"Ollama Model Error: Model '{model_name}' not found at {base_url}. Have you pulled it (e.g., `ollama pull {model_name}`)? Error: {model_e}"
            elif "connection refused" in error_detail: # Should be caught earlier, but double-check
                 error_msg = f"Ollama Connection Error during test invocation: Could not connect to {base_url}. Is Ollama running?"
            else:
                 error_msg = f"Ollama Model Error: Failed during test invocation of model '{model_name}' at {base_url}. The model might be corrupted or incompatible. Check Ollama server logs. Error: {model_e}"
            log_message(error_msg, "ERROR")
            st.error(error_msg)
            return None, None # Fail initialization if test invocation fails

        log_message("Ollama provider initialized successfully.", "INFO")
        return llm, embeddings

    except Exception as e: # Catch errors during LangChain component initialization itself
        error_msg = f"Error initializing Ollama LangChain components: {e}"
        log_message(error_msg, "ERROR")
        st.error(error_msg)
        return None, None

    finally:
        log_message("Ollama initialization process completed.", "DEBUG")
# --- END MODIFIED: _initialize_ollama ---
