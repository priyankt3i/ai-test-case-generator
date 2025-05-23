from typing import Dict, Tuple, Optional

# Langchain Core Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# Google Vertex AI specific imports
try:
    from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
    from google.api_core.exceptions import PermissionDenied as GooglePermissionDenied, ResourceExhausted as GoogleResourceExhausted, NotFound as GoogleNotFound
except ImportError:
    # Fallback for environments where langchain_google_vertexai might not be installed
    ChatVertexAI = type('ChatVertexAI', (object,), {})
    VertexAIEmbeddings = type('VertexAIEmbeddings', (object,), {})
    GooglePermissionDenied = type('GooglePermissionDenied', (Exception,), {})
    GoogleResourceExhausted = type('GoogleResourceExhausted', (Exception,), {})
    GoogleNotFound = type('GoogleNotFound', (Exception,), {})
    print("LLM_VERTEXAI_PROVIDER: WARNING: 'langchain-google-vertexai' not found. Vertex AI functionality may be limited.")

# Import config and utilities
try:
    from config import DEFAULT_TEMPERATURE
    from helper.utils import import_class, log_message
    log_message("Successfully imported config and utils in llm_vertexai.", "DEBUG")
except ImportError as e:
    print(f"LLM_VERTEXAI_PROVIDER: CRITICAL: Failed to import from config or helper.utils: {e}")
    DEFAULT_TEMPERATURE = 0.7
    def log_message(msg, level, **kwargs): print(f"LLM_VERTEXAI_FALLBACK_LOGGER [{level}] {msg}")
    def import_class(mod, cls): return None
    log_message("Using fallback log_message/import_class in llm_vertexai due to import error.", "ERROR")


def _initialize_vertexai(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """
    Initializes Google Vertex AI LLM and Embeddings.
    Requires project_id and location (region) in credentials.
    Authentication typically relies on gcloud CLI, service account, or environment variables.
    """
    log_message("Initializing Vertex AI provider...", "INFO")

    project_id = credentials.get("project_id")
    location = credentials.get("location") # e.g., "us-central1"

    if not project_id or not location:
        log_message("Vertex AI init failed: 'project_id' or 'location' missing in credentials.", "ERROR")
        return None, None

    # For Vertex AI, we'll directly use ChatVertexAI and VertexAIEmbeddings
    # The config_dict should ideally specify these or we can hardcode them here.
    # For consistency with your original code, we'll still use import_class,
    # assuming config_dict will point to 'langchain_google_vertexai.chat_models.ChatVertexAI' etc.
    llm_module_name = config_dict.get("llm_module", "langchain_google_vertexai.chat_models")
    llm_class_name = config_dict.get("llm_class", "ChatVertexAI")
    embeddings_module_name = config_dict.get("embeddings_module", "langchain_google_vertexai.embeddings")
    embeddings_class_name = config_dict.get("embeddings_class", "VertexAIEmbeddings")

    if not all([llm_module_name, llm_class_name, embeddings_module_name, embeddings_class_name]):
        log_message("Vertex AI init failed: Module/class names missing in config_dict.", "ERROR")
        return None, None

    LLMClass = import_class(llm_module_name, llm_class_name)
    EmbeddingsClass = import_class(embeddings_module_name, embeddings_class_name)

    if not LLMClass or not EmbeddingsClass:
        log_message("Vertex AI init failed: Could not import LangChain Vertex AI classes. Ensure `langchain-google-vertexai` is installed.", "ERROR")
        return None, None

    # Default embedding model for Vertex AI
    embed_model_id = config_dict.get("embeddings_model_id", "text-embedding-004")

    try:
        # Initialize ChatVertexAI
        llm = LLMClass(
            model_name=model_name,
            project=project_id,
            location=location,
            temperature=DEFAULT_TEMPERATURE,
            convert_system_message_to_human=True # Vertex AI models like Gemini often benefit from this
        )
        log_message(f"Vertex AI LLM Class {LLMClass.__name__} initialized for model '{model_name}' in project '{project_id}', location '{location}'.", "DEBUG")

        # Initialize VertexAIEmbeddings
        embeddings = EmbeddingsClass(
            model_name=embed_model_id,
            project=project_id,
            location=location
        )
        log_message(f"Vertex AI Embeddings Class {EmbeddingsClass.__name__} initialized (model: {embed_model_id}) in project '{project_id}', location '{location}'.", "DEBUG")

        log_message("Vertex AI provider initialized successfully.", "INFO")
        return llm, embeddings

    except (GooglePermissionDenied, GoogleNotFound, ValueError) as e:
        error_detail = str(e)
        log_message(f"Vertex AI Permission/Not Found/Value Error: {error_detail}", "ERROR")
        if "permission denied" in error_detail.lower():
            log_message(f"Vertex AI Auth Failed: Check project permissions for '{project_id}' and API access for model '{model_name}' or embedding model '{embed_model_id}'.", "ERROR")
        elif "not found" in error_detail.lower():
            log_message(f"Vertex AI Error: Model '{model_name}' or embedding model '{embed_model_id}' not found in location '{location}' for project '{project_id}'.", "ERROR")
        elif "invalid argument" in error_detail.lower():
            log_message(f"Vertex AI Error: Invalid argument provided. Check model name, project ID, or location. Error: {error_detail}", "ERROR")
        return None, None
    except GoogleResourceExhausted as e:
        log_message(f"Vertex AI API Quota Exceeded: {e}", "ERROR")
        return None, None
    except Exception as e:
        error_msg = f"Error initializing Vertex AI: {e}"
        log_message(error_msg, "ERROR", exc_info=True)
        return None, None
    finally:
        log_message("Vertex AI initialization process completed.", "DEBUG")
