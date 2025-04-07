# ll_integration.py
# Handles LangChain setup, LLM interactions (identification, RAG, refactoring).

import streamlit as st
import yaml
import os
import json
import re # Import regex module
from typing import Dict, Any, Tuple, List, Optional
import requests
import inspect # Added for checking function signatures

# Langchain Core Imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# Ollama Imports
try:
    from langchain_ollama import ChatOllama
    from langchain_ollama import OllamaEmbeddings
    LANGCHAIN_OLLAMA_AVAILABLE = True
except ImportError:
    LANGCHAIN_OLLAMA_AVAILABLE = False

# Import config and utilities
try:
    import config
    from config import (
        LLM_PROVIDER_CONFIG, FALLBACK_EMBEDDING_PROVIDERS,
        DEFAULT_TEMPERATURE, APP_CONTEXT_FOLDER_PATH, NO_CONTEXT_OPTION,
        CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_SEARCH_K,
        EXCEL_EXPECTED_COLUMNS
    )
    from utils import import_class, parse_json_output, log_message
except ImportError as e:
    try: log_message(f"CRITICAL: Failed to import required modules (config, utils): {e}", "ERROR")
    except NameError: pass
    st.error(f"CRITICAL: Failed to import required modules (config, utils): {e}")
    st.stop()

# Specific exception imports
try: from botocore.exceptions import NoCredentialsError, ClientError
except ImportError: NoCredentialsError, ClientError = Exception, Exception
try: from openai import AuthenticationError as OpenAIAuthenticationError, RateLimitError as OpenAIRateLimitError
except ImportError: OpenAIAuthenticationError, OpenAIRateLimitError = Exception, Exception
try: from google.api_core.exceptions import PermissionDenied as GooglePermissionDenied, ResourceExhausted as GoogleResourceExhausted
except ImportError: GooglePermissionDenied, GoogleResourceExhausted = Exception, Exception
# Anthropic/Groq might have their own specific exceptions, import if available/needed
try: import boto3 # Ensure boto3 is imported at the top level if used
except ImportError: boto3 = None # Allow the program to run even if boto3 isn't installed, checks happen later


# --- Provider Initialization Helpers ---
def _initialize_openai(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """Initializes OpenAI LLM and Embeddings."""
    log_message("Initializing OpenAI provider...", "INFO")
    api_key = credentials.get("api_key")
    if not api_key:
        log_message("OpenAI init failed: API key missing.", "ERROR")
        st.error("OpenAI Error: API key is missing.")
        return None, None

    LLMClass = import_class(config_dict["llm_module"], config_dict["llm_class"])
    EmbeddingsClass = import_class(config_dict["embeddings_module"], config_dict["embeddings_class"])
    if not LLMClass or not EmbeddingsClass:
        log_message("OpenAI init failed: Could not import required LangChain classes.", "ERROR")
        st.error("Failed to import necessary LangChain classes for OpenAI.")
        return None, None

    try:
        llm = LLMClass(api_key=api_key, model=model_name, temperature=DEFAULT_TEMPERATURE)
        log_message(f"OpenAI LLM Class {LLMClass.__name__} initialized.", "DEBUG")
        embeddings = EmbeddingsClass(api_key=api_key)
        log_message(f"OpenAI Embeddings Class {EmbeddingsClass.__name__} initialized.", "DEBUG")
        log_message("OpenAI provider initialized successfully.", "INFO")
        return llm, embeddings
    except OpenAIAuthenticationError as e:
        log_message(f"OpenAI Authentication Failed: {e}", "ERROR")
        st.error("OpenAI Authentication Failed: Invalid API Key.")
        return None, None
    except OpenAIRateLimitError as e:
        log_message(f"OpenAI Rate Limit Exceeded: {e}", "ERROR")
        st.error("OpenAI Rate Limit Exceeded. Please check your plan or wait.")
        return None, None
    except Exception as e:
        error_msg = f"Error initializing OpenAI: {e}"
        log_message(error_msg, "ERROR")
        if "model_not_found" in str(e).lower():
            st.error(f"OpenAI Error: Model '{model_name}' not found or inaccessible.")
        else:
            st.error(error_msg)
        return None, None

def _initialize_gemini(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """Initializes Gemini LLM and Embeddings."""
    log_message("Initializing Gemini provider...", "INFO")
    api_key = credentials.get("api_key")
    if not api_key:
        log_message("Gemini init failed: API key missing.", "ERROR")
        st.error("Gemini Error: API key is missing.")
        return None, None

    LLMClass = import_class(config_dict["llm_module"], config_dict["llm_class"])
    EmbeddingsClass = import_class(config_dict["embeddings_module"], config_dict["embeddings_class"])
    if not LLMClass or not EmbeddingsClass:
        log_message("Gemini init failed: Could not import required LangChain classes.", "ERROR")
        st.error("Failed to import necessary LangChain classes for Gemini.")
        return None, None

    try:
        llm = LLMClass(google_api_key=api_key, model=model_name, temperature=DEFAULT_TEMPERATURE, convert_system_message_to_human=True)
        log_message(f"Gemini LLM Class {LLMClass.__name__} initialized.", "DEBUG")
        embed_model_id = config_dict.get("embeddings_model_id", "models/embedding-001")
        embeddings = EmbeddingsClass(model=embed_model_id, google_api_key=api_key)
        log_message(f"Gemini Embeddings Class {EmbeddingsClass.__name__} initialized (model: {embed_model_id}).", "DEBUG")
        log_message("Gemini provider initialized successfully.", "INFO")
        return llm, embeddings
    except (GooglePermissionDenied, ValueError) as e:
        log_message(f"Gemini Permission/Value Error: {e}", "ERROR")
        if "api key not valid" in str(e).lower():
            st.error("Gemini Authentication Failed: API Key is not valid. Please check your key and ensure the API is enabled.")
        else:
            st.error(f"Gemini Permission/Value Error: {e}")
        return None, None
    except GoogleResourceExhausted as e:
         log_message(f"Gemini API Quota Exceeded: {e}", "ERROR")
         st.error("Gemini API Quota Exceeded. Please check your usage limits.")
         return None, None
    except Exception as e:
        error_msg = f"Error initializing Gemini: {e}"
        log_message(error_msg, "ERROR")
        st.error(error_msg)
        return None, None

def _initialize_claude(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """Initializes Claude LLM. Embeddings handled by fallback."""
    log_message("Initializing Claude provider...", "INFO")
    api_key = credentials.get("api_key")
    if not api_key:
        log_message("Claude init failed: API key missing.", "ERROR")
        st.error("Claude Error: API key is missing.")
        return None, None

    LLMClass = import_class(config_dict["llm_module"], config_dict["llm_class"])
    if not LLMClass:
        log_message("Claude init failed: Could not import required LangChain class.", "ERROR")
        st.error("Failed to import necessary LangChain class for Claude.")
        return None, None

    try:
        llm = LLMClass(anthropic_api_key=api_key, model=model_name, temperature=DEFAULT_TEMPERATURE)
        log_message(f"Claude LLM Class {LLMClass.__name__} initialized.", "DEBUG")
        log_message("Claude provider initialized successfully (embeddings require fallback).", "INFO")
        return llm, None # Embeddings will use fallback
    except Exception as e:
        error_msg = f"Error initializing Claude: {e}"
        log_message(error_msg, "ERROR")
        if "authentication_error" in str(e).lower():
            st.error(f"Claude Authentication Failed: Check API Key.")
        elif "permission_error" in str(e).lower():
            st.error(f"Claude Permission Error: Key might lack access to model '{model_name}'.")
        elif "invalid_request_error" in str(e).lower() and "model" in str(e).lower():
            st.error(f"Claude Invalid Request: Model '{model_name}' might be incorrect or unavailable.")
        else:
            st.error(error_msg)
        return None, None

# --- MODIFIED: _initialize_bedrock ---
def _initialize_bedrock(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """Initializes AWS Bedrock LLM and Embeddings."""
    log_message("Initializing AWS Bedrock provider...", "INFO")

    # --- Retrieve Credentials ---
    aws_access_key_id = credentials.get("aws_access_key_id")
    aws_secret_access_key = credentials.get("aws_secret_access_key")
    aws_session_token = credentials.get("aws_session_token") # <-- RETRIEVED SESSION TOKEN
    region_name = credentials.get("region_name")
    embedding_model_id = credentials.get("embedding_model_id")

    # --- Basic Credential Presence Check ---
    if not all([aws_access_key_id, aws_secret_access_key, region_name, embedding_model_id]):
        log_message("Bedrock init failed: Missing one or more required base credentials (key, secret, region, embedding model).", "ERROR")
        st.error("AWS Bedrock Error: Missing AWS Access Key ID, Secret Access Key, Region Name, or Embedding Model ID.")
        return None, None

    # --- Session Token Check for Temporary Credentials ---
    # Check if the key looks like a temporary credential (starts with ASIA) and if the session token is missing
    if aws_access_key_id.startswith("ASIA") and not aws_session_token:
         log_message("Bedrock init failed: Temporary credentials (Key ID starts with ASIA) require a Session Token, but it's missing.", "ERROR")
         st.error("AWS Bedrock Error: Using temporary credentials (Access Key ID starts with ASIA), but the AWS Session Token is missing.")
         return None, None

    # --- Boto3 Installation Check ---
    if boto3 is None: # Check if boto3 failed to import earlier
        log_message("Bedrock init failed: boto3 not installed.", "ERROR")
        st.error("AWS Bedrock requires `boto3`. Install it (`pip install boto3`).")
        return None, None

    # --- LangChain Class Import Check ---
    LLMClass = import_class(config_dict["llm_module"], config_dict["llm_class"])
    EmbeddingsClass = import_class(config_dict["embeddings_module"], config_dict["embeddings_class"])
    if not LLMClass or not EmbeddingsClass:
        log_message("Bedrock init failed: Could not import required LangChain classes.", "ERROR")
        st.error("Failed to import necessary LangChain classes for Bedrock.")
        return None, None

    try:
        # --- Prepare Boto3 Client Arguments ---
        # Create a dictionary to hold arguments for boto3.client
        boto3_client_args = {
            'service_name': 'bedrock-runtime',
            'region_name': region_name,
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key
        }
        # Conditionally add the session token ONLY if it was provided
        if aws_session_token:
            boto3_client_args['aws_session_token'] = aws_session_token # <-- ADDED SESSION TOKEN ARG
            log_message("Using AWS Session Token for Bedrock client.", "DEBUG")
        else:
             log_message("No AWS Session Token provided; assuming permanent credentials or environment variables.", "DEBUG")

        # --- Create Boto3 Client ---
        # Use dictionary unpacking (**) to pass the arguments
        bedrock_client = boto3.client(**boto3_client_args)
        log_message(f"Bedrock client created for region {region_name}.", "DEBUG")

        # --- Initialize LangChain LLM ---
        llm_params = {"client": bedrock_client, "model_id": model_name}
        # Handle temperature setting (check if model_kwargs or direct temperature is expected)
        try:
            # Try the modern way with model_kwargs
            llm_params["model_kwargs"] = {"temperature": DEFAULT_TEMPERATURE}
            llm = LLMClass(**llm_params)
        except TypeError:
            # If that fails, try passing temperature directly (might be older LangChain version)
            del llm_params["model_kwargs"]
            # Use inspect to be safer than just assuming 'temperature' is always valid
            sig = inspect.signature(LLMClass)
            if 'temperature' in sig.parameters:
                 llm_params["temperature"] = DEFAULT_TEMPERATURE
                 log_message("Passing temperature directly to LLM class.", "DEBUG")
            else:
                 log_message("LLM class does not accept 'temperature' directly or via 'model_kwargs'. Using default.", "WARNING")
            llm = LLMClass(**llm_params) # Initialize without temperature if not accepted

        log_message(f"Bedrock LLM Class {LLMClass.__name__} initialized (model: {model_name}).", "DEBUG")

        # --- Initialize LangChain Embeddings ---
        embeddings = EmbeddingsClass(client=bedrock_client, model_id=embedding_model_id)
        log_message(f"Bedrock Embeddings Class {EmbeddingsClass.__name__} initialized (model: {embedding_model_id}).", "DEBUG")

        log_message("AWS Bedrock provider initialized successfully.", "INFO")
        return llm, embeddings

    # --- Exception Handling ---
    except NoCredentialsError as e:
        log_message(f"Bedrock credentials error: {e}", "ERROR")
        st.error("AWS Bedrock Error: Credentials not found or invalid. Check configuration/environment.")
        return None, None
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        error_msg_detail = e.response.get('Error', {}).get('Message', str(e))
        log_message(f"Bedrock client error: {error_code} - {error_msg_detail}", "ERROR")
        if error_code == 'AccessDeniedException':
            # Check if it's the specific token error or a general access denied
            if "security token" in error_msg_detail.lower():
                 st.error(f"AWS Bedrock Access Denied: The security token (Session Token) is invalid or expired. Please refresh your credentials.")
            else:
                 st.error(f"AWS Bedrock Access Denied: Check IAM permissions for Bedrock, model '{model_name}', and embedding model '{embedding_model_id}' in region '{region_name}'. Also ensure model access is enabled in Bedrock console.")
        elif error_code == 'ValidationException':
            st.error(f"AWS Bedrock Validation Error: Check region '{region_name}' or model ID '{model_name}' / '{embedding_model_id}'. Details: {error_msg_detail}")
        elif error_code == 'ResourceNotFoundException':
            st.error(f"AWS Bedrock Resource Not Found: Ensure model '{model_name}' or '{embedding_model_id}' is available/enabled in region '{region_name}'.")
        else:
            st.error(f"AWS Bedrock ClientError: {error_code} - {error_msg_detail}")
        return None, None
    except Exception as e:
        error_msg = f"Error initializing AWS Bedrock: {e}"
        log_message(error_msg, "ERROR")
        st.error(error_msg)
        return None, None
# --- END MODIFIED: _initialize_bedrock ---

def _initialize_groq(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """Initializes Groq LLM. Embeddings handled by fallback."""
    log_message("Initializing Groq provider...", "INFO")
    api_key = credentials.get("api_key")
    if not api_key:
        log_message("Groq init failed: API key missing.", "ERROR")
        st.error("Groq Error: API key is missing.")
        return None, None

    LLMClass = import_class(config_dict["llm_module"], config_dict["llm_class"])
    if not LLMClass:
        log_message("Groq init failed: Could not import required LangChain class.", "ERROR")
        st.error("Failed to import necessary LangChain class for Groq.")
        return None, None

    try:
        llm = LLMClass(groq_api_key=api_key, model_name=model_name, temperature=DEFAULT_TEMPERATURE)
        log_message(f"Groq LLM Class {LLMClass.__name__} initialized.", "DEBUG")
        log_message("Groq provider initialized successfully (embeddings require fallback).", "INFO")
        return llm, None # Embeddings will use fallback
    except Exception as e:
        error_msg = f"Error initializing Groq: {e}"
        log_message(error_msg, "ERROR")
        if "authentication" in str(e).lower(): st.error(f"Groq Authentication Failed: Check API Key.")
        elif "invalid_request" in str(e).lower() and "model" in str(e).lower(): st.error(f"Groq Invalid Request: Model '{model_name}' might be incorrect or unavailable.")
        else: st.error(error_msg)
        return None, None

def _get_fallback_embeddings(fallback_openai_key: str) -> Optional[Embeddings]:
    """Initializes OpenAI embeddings for fallback."""
    log_message("Attempting to initialize OpenAI fallback embeddings...", "INFO")
    if not fallback_openai_key:
        log_message("OpenAI fallback init failed: API key missing.", "ERROR")
        st.error("RAG requires an OpenAI API key for fallback embeddings, but it's missing.")
        return None
    EmbeddingsClass = import_class("langchain_openai", "OpenAIEmbeddings")
    if not EmbeddingsClass:
        log_message("OpenAI fallback init failed: Could not import OpenAIEmbeddings.", "ERROR")
        st.error("Failed to import OpenAIEmbeddings for fallback.")
        return None
    try:
        embeddings = EmbeddingsClass(api_key=fallback_openai_key)
        log_message("OpenAI fallback embeddings initialized successfully.", "INFO")
        return embeddings
    except OpenAIAuthenticationError as e:
        log_message(f"OpenAI Fallback Auth Error: {e}", "ERROR")
        st.error("OpenAI Fallback Auth Error: Invalid API Key provided for fallback embeddings.")
        return None
    except Exception as e:
        log_message(f"Error initializing OpenAI fallback embeddings: {e}", "ERROR")
        st.error(f"Error initializing OpenAI fallback embeddings: {e}")
        return None

def _initialize_ollama(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """Initializes Ollama LLM and Embeddings."""
    log_message("Initializing Ollama provider...", "INFO")
    if not LANGCHAIN_OLLAMA_AVAILABLE:
        log_message("Ollama init failed: langchain-ollama package not available.", "ERROR")
        st.error("Ollama integration requires the `langchain-ollama` package. Please install it (`pip install langchain-ollama`).")
        return None, None

    base_url = credentials.get("base_url", "http://localhost:11434").strip()
    if not base_url:
        base_url = "http://localhost:11434"
        log_message("Ollama Base URL empty, using default: http://localhost:11434", "WARNING")

    if not model_name:
        log_message("Ollama init failed: Model name missing.", "ERROR")
        st.error("Ollama requires a model to be selected.")
        return None, None
    log_message(f"Ollama using Base URL: {base_url}, Model: {model_name}", "DEBUG")

    # Check Ollama server reachability
    try:
        response = requests.get(base_url, timeout=5)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        log_message(f"Successfully connected to Ollama server at {base_url}", "DEBUG")
    except requests.exceptions.ConnectionError:
        error_msg = f"Ollama Connection Error: Could not connect to server at {base_url}. Is Ollama running?"
        log_message(error_msg, "ERROR")
        st.error(error_msg)
        return None, None
    except requests.exceptions.Timeout:
        error_msg = f"Ollama Connection Timeout: Server at {base_url} did not respond quickly enough."
        log_message(error_msg, "ERROR")
        st.error(error_msg)
        return None, None
    except requests.exceptions.RequestException as e: # Catch other request errors (like HTTPError)
        error_msg = f"Ollama Request Error: Failed to query {base_url}. Status Code: {e.response.status_code if e.response else 'N/A'}. Error: {e}"
        log_message(error_msg, "ERROR")
        st.error(error_msg)
        return None, None

    # Initialize LangChain components
    try:
        llm = ChatOllama(
            base_url=base_url,
            model=model_name,
            temperature=DEFAULT_TEMPERATURE
        )
        log_message(f"Ollama LLM Class ChatOllama initialized.", "DEBUG")

        # Use the specified model for embeddings by default, could be made configurable
        embeddings_model = model_name
        embeddings = OllamaEmbeddings(
            base_url=base_url,
            model=embeddings_model
        )
        log_message(f"Ollama Embeddings Class OllamaEmbeddings initialized (using model: {embeddings_model}).", "DEBUG")

        # Test model invocation to catch model-specific errors early
        try:
            # Use a simple, non-empty prompt
            llm.invoke("Hi, how are you?")
            log_message(f"Ollama model '{model_name}' responded successfully.", "DEBUG")
        except Exception as model_e:
            # Improve error message for common issues
            error_detail = str(model_e)
            if "model not found" in error_detail.lower():
                 error_msg = f"Ollama Model Error: Model '{model_name}' not found at {base_url}. Have you pulled it (e.g., `ollama pull {model_name}`)? Error: {model_e}"
            else:
                 error_msg = f"Ollama Model Error: Failed to invoke model '{model_name}' at {base_url}. Is it running correctly? Error: {model_e}"
            log_message(error_msg, "ERROR")
            st.error(error_msg)
            return None, None

        log_message("Ollama provider initialized successfully.", "INFO")
        return llm, embeddings
    except Exception as e: # Catch errors during LangChain component initialization
        error_msg = f"Error initializing Ollama LangChain components: {e}"
        log_message(error_msg, "ERROR")
        st.error(error_msg)
        return None, None


# --- Main Initialization Function ---
def get_llm_and_embeddings(provider: str, model_name: str, credentials: Dict, fallback_openai_key: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """Initializes LangChain LLM and Embeddings objects based on the selected provider."""
    log_message(f"Getting LLM and Embeddings for provider: {provider}, model: {model_name}", "INFO")
    llm: Optional[BaseChatModel] = None
    embeddings: Optional[Embeddings] = None
    config_dict = LLM_PROVIDER_CONFIG.get(provider)

    if not config_dict:
        log_message(f"Invalid provider selected: {provider}", "ERROR")
        st.error(f"Invalid provider selected: {provider}")
        return None, None

    is_fallback_provider = provider in FALLBACK_EMBEDDING_PROVIDERS
    log_message(f"Is fallback provider? {is_fallback_provider}", "DEBUG")

    # Map provider names to their initialization functions
    init_functions = {
        "OpenAI": _initialize_openai,
        "Gemini": _initialize_gemini,
        "Claude": _initialize_claude,
        "AWS Bedrock": _initialize_bedrock,
        "Groq": _initialize_groq,
        "Ollama": _initialize_ollama,
    }
    init_func = init_functions.get(provider)

    if not init_func:
        log_message(f"Initialization logic not defined for provider: {provider}", "ERROR")
        st.error(f"Initialization logic not defined for provider: {provider}")
        return None, None

    # Call the appropriate initialization function
    llm, embeddings = init_func(config_dict, credentials, model_name)

    # Handle fallback embeddings if the provider needs them and LLM init succeeded
    if is_fallback_provider and llm and not embeddings:
        log_message(f"Provider {provider} requires fallback embeddings.", "INFO")
        embeddings = _get_fallback_embeddings(fallback_openai_key)
        if not embeddings:
            log_message(f"Failed to initialize fallback embeddings for {provider}.", "WARNING")
            st.warning(f"RAG may fail for {provider} as fallback embeddings (using OpenAI) could not be initialized. Check fallback API key.")
            # For fallback providers, we might allow proceeding with just the LLM
            return llm, None

    # Final checks after initialization attempts
    if not llm:
        log_message(f"LLM initialization final check failed for {provider}.", "ERROR")
        # Error message should have already been shown by the specific init helper
        return None, None # If LLM fails, return None for both

    if not embeddings:
        # Non-fallback providers (OpenAI, Gemini, Bedrock, Ollama) *should* have their own embeddings
        if not is_fallback_provider:
            log_message(f"Embeddings initialization final check failed for non-fallback provider {provider}. RAG will not work.", "ERROR")
            st.error(f"Embeddings initialization failed for {provider}. RAG features will not work.")
            # Return the LLM but indicate embeddings failed
            return llm, None
        # If it IS a fallback provider and embeddings failed (even fallback), we already warned.
        # We return llm, None as decided above.

    log_message(f"LLM and Embeddings initialized successfully for {provider}.", "INFO")
    return llm, embeddings


# --- Credential Checking ---
def check_credentials(provider: str, credentials: Dict, fallback_key: str, require_fallback_for_rag: bool) -> Tuple[bool, str]:
    """Validates if all necessary credentials are present for the selected provider."""
    log_message(f"Checking credentials for provider: {provider}", "DEBUG")
    config_dict = LLM_PROVIDER_CONFIG.get(provider)
    if not config_dict: return False, "Invalid provider selected."

    missing = []
    required_creds = config_dict.get("credentials", [])

    for key in required_creds:
        # Special handling for Ollama: base_url can be empty (use default), model comes from selection later
        if provider == "Ollama" and key in ["base_url", "model"]:
            continue
        # Special handling for Bedrock session token: Only strictly required if key ID starts with ASIA
        if provider == "AWS Bedrock" and key == "aws_session_token":
            # Check if key ID suggests temporary creds but token is missing
            key_id = credentials.get("aws_access_key_id", "")
            token = credentials.get("aws_session_token", "")
            if key_id.startswith("ASIA") and not token:
                 missing.append("AWS Session Token (Required for temporary credentials)")
            continue # Don't do the standard check below for session token

        # Standard check for other required credentials
        if not credentials.get(key, "").strip():
            # Make the key name more user-friendly
            friendly_key_name = key.replace("_", " ").title()
            missing.append(friendly_key_name)

    # Check for fallback key requirement if RAG is intended
    needs_fallback = provider in FALLBACK_EMBEDDING_PROVIDERS
    if require_fallback_for_rag and needs_fallback and not fallback_key.strip():
        missing.append("OpenAI API Key (Required for RAG Fallback Embeddings)")

    if missing:
        error_msg = f"Missing credentials/settings for {provider}: {', '.join(missing)}."
        log_message(error_msg, "WARNING")
        return False, error_msg

    log_message(f"Credential check passed for {provider}.", "DEBUG")
    return True, ""


# --- Prompt Template Fetching ---
def get_prompt_template(provider_name: str, template_key: str) -> str:
    """
    Gets the appropriate prompt template for the given provider and key.
    Falls back to the default template if no provider-specific override exists.

    Args:
        provider_name: The name of the selected LLM provider (e.g., "OpenAI", "Ollama").
        template_key: The key for the desired template
                      (e.g., "IDENTIFY_APP", "GENERATE_TC", "REFACTOR_TC").

    Returns:
        The prompt template string.

    Raises:
        KeyError: If the default template key is not found in config.
        AttributeError: If the default template name isn't found in the config module.
    """
    # Ensure config module is accessible
    if 'config' not in globals():
         raise ImportError("Configuration module 'config' not loaded.")

    provider_config = config.LLM_PROVIDER_CONFIG.get(provider_name, {})
    provider_prompts = provider_config.get("prompt_templates", {})

    # 1. Check for provider-specific override first
    if template_key in provider_prompts:
        log_message(f"Using '{template_key}' prompt override for provider '{provider_name}'", "DEBUG")
        return provider_prompts[template_key]

    # 2. Fallback to default template in config module
    # Construct the expected variable name in config.py (e.g., IDENTIFY_APP_PROMPT_TEMPLATE)
    default_template_name = f"{template_key}_PROMPT_TEMPLATE"
    if hasattr(config, default_template_name):
        log_message(f"Using default '{template_key}' prompt for provider '{provider_name}'", "DEBUG")
        return getattr(config, default_template_name)
    else:
        # This indicates a setup error in config.py
        error_msg = f"Default prompt template variable '{default_template_name}' not found in config.py for key '{template_key}'"
        log_message(error_msg, "ERROR")
        # Raise an error that's more indicative of the problem
        raise AttributeError(error_msg)


# --- Helper: Extract list-like content from LLM output ---
def _extract_list_from_llm_output(raw_output: str) -> Optional[str]:
    """
    Attempts to extract a Python list string ('[...]') or reconstruct one
    from bullet points in potentially messy LLM output.

    Args:
        raw_output: The raw string output from the LLM.

    Returns:
        A string that looks like a Python list ('[...]'), or None if extraction fails.
    """
    log_message("Attempting to extract list structure from raw LLM output...", "DEBUG")
    if not raw_output: return None # Handle empty input

    # 1. Try to find a direct Python list literal '[...]' using regex
    # Handles lists spanning multiple lines with re.DOTALL
    # Makes the regex less greedy (.*?) to find the *first* valid list
    list_match = re.search(r'(\[.*?\])', raw_output, re.DOTALL)
    if list_match:
        extracted = list_match.group(1).strip()
        log_message(f"Found potential direct list literal: '{extracted[:100]}...'", "DEBUG")
        # Basic validation: does it start/end with brackets?
        if extracted.startswith('[') and extracted.endswith(']'):
            # Attempt a quick JSON parse to verify structure - more robust
            try:
                json.loads(extracted)
                log_message("Direct list literal confirmed via JSON parsing.", "DEBUG")
                return extracted
            except json.JSONDecodeError:
                log_message("Regex matched brackets but content is not valid JSON list.", "WARNING")
                # Continue to bullet point check
        else:
            log_message("Regex matched brackets but result seems invalid (doesn't start/end correctly).", "WARNING")

    # 2. If no direct list found or validation failed, try parsing bullet points (* or -)
    log_message("Trying bullet point extraction.", "DEBUG")
    bullet_items = []
    # Regex to find lines starting with optional whitespace, then '*' or '-', then whitespace, then capture content
    bullet_pattern = re.compile(r'^\s*[\*\-]\s+(.*)', re.MULTILINE)
    matches = bullet_pattern.findall(raw_output)

    if matches:
        log_message(f"Found {len(matches)} potential bullet point items.", "DEBUG")
        for item in matches:
            # Clean the item: remove leading/trailing whitespace and quotes
            cleaned_item = item.strip().strip("'\"`") # Added backtick removal
            # Escape internal single quotes for the final string representation
            escaped_item = cleaned_item.replace("'", "\\'")
            if escaped_item: # Avoid adding empty strings
                # Represent as a Python string literal within the list
                bullet_items.append(f"'{escaped_item}'")

        if bullet_items:
            # Construct the Python list string
            list_string = f"[{', '.join(bullet_items)}]"
            log_message(f"Constructed list from bullets: {list_string}", "DEBUG")
            # Verify the constructed string is valid JSON
            try:
                json.loads(list_string)
                log_message("Constructed list from bullets confirmed via JSON parsing.", "DEBUG")
                return list_string
            except json.JSONDecodeError:
                 log_message("Constructed list from bullets is not valid JSON.", "WARNING")
                 # Fall through, maybe raw output is better
        else:
            log_message("Found bullet markers but no valid content extracted.", "DEBUG")

    # 3. If neither method worked
    log_message("Could not extract a valid list structure using regex or bullet points.", "WARNING")
    return None # Indicate that extraction failed


# --- LLM Interaction Functions ---

# --- UPDATED: identify_applications to use _extract_list_from_llm_output ---
def identify_applications(text: str, llm: BaseChatModel, provider_name: str) -> List[str]:
    """
    Identifies application names from text using the provided LLM and provider-specific prompt.
    Includes post-processing to extract list data from potentially messy output before parsing.

    Args:
        text: The input text (e.g., from requirements doc).
        llm: The initialized LangChain Chat Model.
        provider_name: The name of the LLM provider being used.

    Returns:
        A list of identified application names, or an empty list on failure.
    """
    log_message(f"Starting application identification using provider: {provider_name}...", "INFO")
    # --- Input validation ---
    if not text or not text.strip():
        log_message("Identification failed: Input text is empty or whitespace.", "ERROR")
        st.error("Cannot identify applications: Input text is empty.")
        return []
    if not llm:
         log_message("Identification failed: LLM is not initialized.", "ERROR")
         st.error("Cannot identify applications: LLM is not initialized.")
         return []

    try:
        # --- Get Prompt and Build Chain ---
        template_key = "IDENTIFY_APP"
        prompt_template_str = get_prompt_template(provider_name, template_key)
        app_prompt = ChatPromptTemplate.from_template(prompt_template_str)
        log_message(f"Using prompt for app identification:\n{prompt_template_str}", "DEBUG") # Log the prompt
        app_chain = app_prompt | llm | StrOutputParser()
        log_message("Application identification chain created.", "DEBUG")

        # --- Invoke LLM ---
        with st.spinner(f"Asking LLM ({llm.__class__.__name__}) to identify applications..."):
            result_str = app_chain.invoke({"text": text})
            log_message("LLM invocation for identification complete.", "DEBUG")
            log_message(f"Raw LLM output for identification:\n---\n{result_str}\n---", "DEBUG")

        if not result_str or not result_str.strip():
             log_message("LLM returned empty response for identification.", "WARNING")
             st.warning("LLM returned an empty response. Could not identify applications.")
             return []

        # --- Attempt to extract list structure BEFORE parsing ---
        extracted_list_str = _extract_list_from_llm_output(result_str)
        string_to_parse = result_str # Default to raw output if extraction fails

        if extracted_list_str:
            log_message("Using extracted list string for parsing.", "INFO")
            string_to_parse = extracted_list_str
        else:
            log_message("Extraction failed, attempting to parse raw LLM output directly.", "WARNING")
            # Consider a warning if extraction fails often, but might be noisy
            # st.warning("Could not cleanly extract list structure, parsing raw output.")

        # --- Parse the (potentially extracted) result ---
        log_message(f"Attempting to parse string for app list: '{string_to_parse[:200]}...'", "DEBUG")
        # Use the robust parse_json_output which handles errors and logs them
        parsed_apps = parse_json_output(string_to_parse, expected_type=list)

        # --- Process Parsing Result ---
        if parsed_apps is None:
            log_message("Failed to parse application list (after potential extraction).", "WARNING")
            # parse_json_output already shows st.error, but we can add context
            st.warning("LLM response for applications was not in the expected list format. Trying fallback.")

            # --- Fallback Parsing: Comma-separated (only if extraction failed) ---
            # Only try comma fallback if extraction *also* failed and raw output is simple
            if extracted_list_str is None and result_str and not result_str.strip().startswith(("[", "{")) and ',' in result_str:
                # Split by comma, strip whitespace and quotes
                possible_apps = [app.strip().strip("'\"`") for app in result_str.split(',') if app.strip()]
                if possible_apps:
                    log_message(f"Attempting basic comma parsing as final fallback. Found: {possible_apps}", "INFO")
                    st.info("LLM output wasn't structured, attempting basic comma parsing.")
                    # Remove duplicates and sort
                    return sorted(list(set(possible_apps)))

            # If parsing failed even after extraction, or fallback doesn't apply/work
            st.error("Could not reliably identify applications from the LLM response format.")
            return []

        # --- Success Case (parsed_apps is a list) ---
        # Clean the list: ensure all items are strings and remove empty/whitespace ones
        cleaned_apps = [str(app).strip() for app in parsed_apps if isinstance(app, (str, int, float)) and str(app).strip()]
        if not cleaned_apps:
             log_message("Parsed list was empty or contained only non-string/empty items.", "WARNING")
             st.warning("LLM identified an empty list of applications.")
             return []

        # Remove duplicates and sort
        final_apps = sorted(list(set(cleaned_apps)))
        log_message(f"Identification successful. Found apps: {final_apps}", "INFO")
        return final_apps

    # --- Handle Exceptions during the process ---
    except Exception as e:
        log_message(f"Exception during application identification: {type(e).__name__} - {e}", "ERROR")
        st.error(f"An error occurred during application identification LLM call: {e}")
        # Optionally re-raise or handle specific exceptions if needed
        return []


# --- generate_test_cases Function ---
def generate_test_cases(
    text: str,
    selected_apps: List[str],
    context_selections: Dict[str, str],
    llm: BaseChatModel,
    embeddings: Embeddings,
    provider_name: str
) -> Dict[str, Any]:
    """
    Generates test cases for selected applications using RAG and provider-specific prompt.
    Includes filtering of parsed list items to ensure they are dictionaries.

    Args:
        text: The original requirements text.
        selected_apps: List of application names to generate cases for.
        context_selections: Dict mapping app names to selected context file base names.
        llm: Initialized LangChain Chat Model.
        embeddings: Initialized LangChain Embeddings model.
        provider_name: The name of the LLM provider being used.

    Returns:
        A dictionary where keys are app names and values are either a list
        of generated test case dicts or an error string.
    """
    log_message(f"--- Entered generate_test_cases using provider: {provider_name} ---", "DEBUG")
    results = {}

    # --- Initial checks ---
    if not selected_apps:
        log_message("Generation skipped: No applications selected.", "WARNING")
        st.warning("No applications selected for test case generation.")
        return results
    if not text or not text.strip():
        log_message("Generation failed: Source text is empty.", "ERROR")
        st.error("Cannot generate test cases: Source text is empty.")
        return {app: "Error: Source text is empty." for app in selected_apps}
    if not llm:
         log_message("Generation failed: LLM is not initialized.", "ERROR")
         st.error("Cannot generate test cases: LLM is not initialized.")
         return {app: "Error: LLM not initialized." for app in selected_apps}
    if not embeddings:
        log_message("Generation failed: Embeddings are not initialized (required for RAG).", "ERROR")
        st.error("Cannot generate test cases: Embeddings are not initialized (required for RAG).")
        return {app: "Error: Embeddings not initialized." for app in selected_apps}

    # 1. Create Vector Store
    vectorstore = None
    try:
        log_message("--- Attempting Vector Store Creation ---", "DEBUG")
        embedding_source_name = embeddings.__class__.__name__
        log_message(f"Using embeddings: {embedding_source_name}", "DEBUG")
        with st.spinner(f"Creating text embeddings using {embedding_source_name}..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            splits = text_splitter.split_text(text)
            log_message(f"Split text into {len(splits)} chunks.", "DEBUG")
            if not splits:
                log_message("Text splitting resulted in zero chunks.", "ERROR")
                st.error("Text splitting resulted in zero chunks. Cannot create vector store.")
                return {app: "Error: Text splitting failed." for app in selected_apps}
            # Ensure embeddings object is valid before proceeding
            if not hasattr(embeddings, 'embed_documents'):
                 log_message("Embeddings object is invalid (missing embed_documents method).", "ERROR")
                 st.error("Embeddings object is invalid. Cannot create vector store.")
                 return {app: "Error: Invalid embeddings object." for app in selected_apps}

            vectorstore = FAISS.from_texts(texts=splits, embedding=embeddings)
        log_message("--- Vector Store Creation Succeeded ---", "INFO")
    except Exception as e:
        log_message(f"--- Exception during Vector Store Creation: {type(e).__name__} - {e} ---", "ERROR")
        # Provide more specific error message if possible
        st.error(f"Error creating vector store/embeddings: {e}")
        return {app: f"Error creating embeddings: {e}" for app in selected_apps}

    # This check should be redundant if the try/except works, but belt-and-suspenders
    if vectorstore is None:
        log_message("Vector store object is None after creation block (unexpected).", "ERROR")
        st.error("Cannot proceed: Vector store initialization failed unexpectedly.")
        return {app: "Error: Vector store initialization failed unexpectedly." for app in selected_apps}

    # 2. Setup RAG Chain
    retrieval_chain = None
    try:
        log_message("--- Entering RAG Setup ---", "DEBUG")
        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_SEARCH_K})
        log_message(f"Retriever Type: {type(retriever)}", "DEBUG")

        # Get and format the prompt template
        template_key = "GENERATE_TC"
        prompt_template_str = get_prompt_template(provider_name, template_key)
        # Ensure expected columns are available from config
        if not EXCEL_EXPECTED_COLUMNS or not isinstance(EXCEL_EXPECTED_COLUMNS, list):
             raise ValueError("EXCEL_EXPECTED_COLUMNS not defined correctly in config.")
        tc_fields = ", ".join([f"`{col}`" for col in EXCEL_EXPECTED_COLUMNS])
        # Use try-except for formatting in case the template is wrong
        try:
            formatted_prompt_str = prompt_template_str.format(field_names=tc_fields)
        except KeyError as ke:
             log_message(f"Prompt template format error: Missing key {ke}", "ERROR")
             st.error(f"Prompt template '{template_key}' is missing the required placeholder: {{field_names}}")
             raise ValueError(f"Prompt template format error: Missing key {ke}")

        test_case_prompt = ChatPromptTemplate.from_template(formatted_prompt_str)
        log_message(f"Using prompt for TC generation:\n{formatted_prompt_str}", "DEBUG")

        # Create the chains
        log_message("Attempting create_stuff_documents_chain...", "DEBUG")
        document_chain = create_stuff_documents_chain(llm, test_case_prompt)
        log_message("create_stuff_documents_chain Succeeded.", "DEBUG")

        log_message("Attempting create_retrieval_chain...", "DEBUG")
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        log_message("create_retrieval_chain Succeeded.", "INFO")

    except Exception as e:
        log_message(f"--- Exception during RAG Setup: {type(e).__name__} - {e} ---", "ERROR")
        st.error(f"Fatal error setting up RAG chain components: {e}")
        # Ensure all selected apps get an error message if setup fails
        error_message_for_results = f"Error setting up RAG chain: {e}"
        for app in selected_apps:
            if app not in results: # Avoid overwriting specific errors if any occurred before this point
                results[app] = error_message_for_results
        return results

    # Check if chain creation succeeded before proceeding
    if not retrieval_chain:
         log_message("Cannot proceed with generation as RAG chain setup failed (retrieval_chain is None).", "ERROR")
         st.error("Cannot proceed with generation as RAG chain setup failed.")
         # Return results which should contain the setup error message from the except block
         return results

    # 3. Generate for each selected application
    total_selected_apps = len(selected_apps)
    log_message(f"Starting generation loop for {total_selected_apps} selected apps.", "INFO")
    progress_bar = st.progress(0.0, text="Initializing test case generation...")

    for i, app_name in enumerate(selected_apps):
        log_message(f"Processing app {i+1}/{total_selected_apps}: {app_name}", "INFO")
        # Update progress
        progress_value = (i + 1) / total_selected_apps
        progress_text = f"Generating for '{app_name}' ({i+1}/{total_selected_apps})..."
        progress_bar.progress(progress_value, text=progress_text)

        # Use st.status for collapsible progress per app
        with st.status(f"Processing '{app_name}'...", expanded=False) as status:
            try:
                # --- Context File Loading ---
                yaml_context_str = ""
                selected_yaml_base = context_selections.get(app_name, NO_CONTEXT_OPTION)
                log_message(f"App '{app_name}': Selected context file base = '{selected_yaml_base}'", "DEBUG")
                status.write(f"Selected Context File: '{selected_yaml_base}'")

                if selected_yaml_base != NO_CONTEXT_OPTION:
                    # Construct potential paths (.yaml and .yml)
                    yaml_filename_yaml = f"{selected_yaml_base}.yaml"
                    yaml_path_yaml = os.path.join(APP_CONTEXT_FOLDER_PATH, yaml_filename_yaml)
                    yaml_filename_yml = f"{selected_yaml_base}.yml"
                    yaml_path_yml = os.path.join(APP_CONTEXT_FOLDER_PATH, yaml_filename_yml)

                    yaml_path_to_load = None
                    actual_filename = None
                    # Check which extension exists
                    if os.path.exists(yaml_path_yaml):
                        yaml_path_to_load = yaml_path_yaml
                        actual_filename = yaml_filename_yaml
                    elif os.path.exists(yaml_path_yml):
                        yaml_path_to_load = yaml_path_yml
                        actual_filename = yaml_filename_yml

                    if yaml_path_to_load:
                        log_message(f"App '{app_name}': Loading context from {yaml_path_to_load}", "DEBUG")
                        try:
                            with open(yaml_path_to_load, 'r', encoding='utf-8') as yf:
                                yaml_data = yaml.safe_load(yf)
                                # Format the loaded YAML nicely for the prompt
                                yaml_context_str = f"\n\n--- Additional Context ({actual_filename}) ---\n{yaml.dump(yaml_data, indent=2, allow_unicode=True, sort_keys=False)}\n--- End Context ---"
                            log_message(f"App '{app_name}': Context loaded successfully.", "DEBUG")
                            status.write(f"Successfully loaded context from {actual_filename}")
                        except yaml.YAMLError as ye:
                            status.warning(f"⚠️ Error parsing YAML file {actual_filename}: {ye}")
                            log_message(f"App '{app_name}': YAML parse error in {actual_filename} - {ye}", "WARNING")
                        except OSError as oe:
                            status.warning(f"⚠️ Error reading YAML file {actual_filename}: {oe}")
                            log_message(f"App '{app_name}': YAML read error for {actual_filename} - {oe}", "WARNING")
                        except Exception as e: # Catch any other unexpected errors during loading/dumping
                            status.warning(f"⚠️ Unexpected error loading/processing context {actual_filename}: {e}")
                            log_message(f"App '{app_name}': Unexpected YAML load/dump error for {actual_filename} - {e}", "WARNING")
                    else:
                        # Only warn if a specific file was selected but not found
                        status.warning(f"⚠️ Context file '{selected_yaml_base}.yaml/.yml' not found in '{APP_CONTEXT_FOLDER_PATH}'.")
                        log_message(f"App '{app_name}': Context file '{selected_yaml_base}.yaml/.yml' not found.", "WARNING")
                else:
                     log_message(f"App '{app_name}': No additional context file selected.", "DEBUG")


                # --- Prepare Input and Invoke Chain ---
                # Construct the input query for the RAG chain
                input_query_string = f"Generate test cases specifically relevant to the application or system named: '{app_name}'. Use the retrieved requirements context and the additional context provided below (if any) to inform the test cases.{yaml_context_str}"
                log_message(f"App '{app_name}': Prepared input query for RAG.", "DEBUG")
                # Log the full query only if debugging level allows (can be long)
                # log_message(f"App '{app_name}': Full RAG Input Query:\n{input_query_string}", "TRACE") # Assuming TRACE level

                status.write("Invoking RAG chain...")
                log_message(f"App '{app_name}': Invoking retrieval_chain...", "DEBUG")
                # Invoke the RAG chain
                response = retrieval_chain.invoke({"input": input_query_string})
                log_message(f"App '{app_name}': Received response from retrieval_chain.", "DEBUG")
                status.write("Received response from LLM.")

                # --- Process Response ---
                # Check if the response structure is as expected (dict with 'answer')
                if isinstance(response, dict) and "answer" in response and response["answer"] and isinstance(response["answer"], str):
                    answer_str = response["answer"].strip()
                    log_message(f"App '{app_name}': Got answer string from response dict. Length: {len(answer_str)}", "DEBUG")
                    log_message(f"App '{app_name}': Raw Answer:\n---\n{answer_str}\n---", "DEBUG") # Log raw answer

                    # Use the robust JSON parser
                    parsed_cases = parse_json_output(answer_str, expected_type=list)

                    if parsed_cases is None:
                        # parse_json_output already showed error and logged
                        results[app_name] = "Error: Failed to parse JSON list from LLM response."
                        # Update status to reflect the error
                        status.update(label="⚠️ JSON Parse Error", state="error", expanded=True)
                    elif isinstance(parsed_cases, list):
                        # Filter out any items in the list that are not dictionaries
                        valid_cases = [item for item in parsed_cases if isinstance(item, dict)]
                        log_message(f"App '{app_name}': Parsed {len(parsed_cases)} items, filtered to {len(valid_cases)} valid dicts.", "DEBUG")

                        if not valid_cases:
                            # If the list is empty after filtering
                            results[app_name] = "Error: LLM response parsed as a list, but contained no valid test case objects (dictionaries)."
                            log_message(f"App '{app_name}': Parsed list contained no valid dictionary items.", "ERROR")
                            status.update(label="⚠️ No Valid Cases Found", state="error", expanded=True)
                        else:
                            # Success: We have at least one valid dictionary (test case)
                            results[app_name] = valid_cases # Assign the filtered list
                            log_message(f"App '{app_name}': Successfully extracted {len(valid_cases)} valid test cases.", "INFO")
                            status.update(label=f"✓ Generated {len(valid_cases)} cases", state="complete")
                            # Optional: Log/warn if some items were filtered out
                            if len(valid_cases) < len(parsed_cases):
                                filtered_count = len(parsed_cases) - len(valid_cases)
                                log_message(f"App '{app_name}': Filtered out {filtered_count} non-dictionary items from LLM response.", "WARNING")
                                status.warning(f"Note: {filtered_count} non-test case elements were filtered from the LLM's response.")
                    else:
                        # This case should ideally not be reached if parse_json_output works correctly
                         results[app_name] = f"Error: Unexpected parsing result type '{type(parsed_cases).__name__}'."
                         log_message(f"App '{app_name}': Unexpected parse result type {type(parsed_cases).__name__}.", "ERROR")
                         status.update(label="⚠️ Parse Type Error", state="error", expanded=True)

                else: # Handle case where response is not dict or 'answer' key is missing/empty/not string
                    results[app_name] = "Error: LLM provided no answer or unexpected response structure."
                    log_message(f"App '{app_name}': No 'answer' key in response or response structure invalid. Response: {response}", "ERROR")
                    status.update(label="⚠️ No Answer/Bad Format", state="error", expanded=True)

            except Exception as e:
                # Catch-all for errors during the processing of a single app
                log_message(f"--- Exception during Generation Loop for '{app_name}': {type(e).__name__} - {e} ---", "ERROR")
                st.error(f"An error occurred during generation for '{app_name}': {e}")
                results[app_name] = f"Error: Generation failed - {e}"
                # Update status to failed
                status.update(label="❌ Failed", state="error", expanded=True)

    # Clear the overall progress bar
    progress_bar.empty()
    log_message("--- Finished generate_test_cases ---", "DEBUG")
    return results


# --- refactor_single_test_case Function ---
def refactor_single_test_case(
    app_name: str,
    tc_id: str,
    instructions: str,
    original_tc_data: Dict,
    llm: BaseChatModel,
    provider_name: str
) -> Optional[Dict]:
    """Uses LLM to refactor a single test case based on instructions and provider-specific prompt."""
    log_message(f"Starting refactor for TC '{tc_id}' in app '{app_name}' using provider {provider_name}...", "INFO")

    # --- Input Validation ---
    if not llm:
        st.error("Cannot refactor: LLM is not initialized.")
        log_message("Refactor failed: LLM not initialized.", "ERROR")
        return None
    if not original_tc_data or not isinstance(original_tc_data, dict):
        st.error(f"Cannot refactor: Invalid original test case data provided for TC ID '{tc_id}'.")
        log_message(f"Refactor failed: Invalid original data for TC '{tc_id}'. Type: {type(original_tc_data)}", "ERROR")
        return None
    if not instructions or not instructions.strip():
         # Changed to info as it's user action, not system error
         st.info(f"Refactor skipped for TC '{tc_id}': Modification instructions are empty.")
         log_message(f"Refactor skipped: Empty instructions for TC '{tc_id}'.", "INFO")
         return None # Return None, indicating no change/action

    try:
        # --- Prepare Inputs ---
        # Safely serialize original data to JSON string
        try:
            original_json_str = json.dumps(original_tc_data, indent=2)
        except TypeError as te:
             log_message(f"Refactor failed for TC '{tc_id}': Could not serialize original data to JSON: {te}", "ERROR")
             st.error(f"Error preparing original test case '{tc_id}' for refactoring: {te}")
             return None

        # --- Get Prompt and Build Chain ---
        template_key = "REFACTOR_TC"
        prompt_template_str = get_prompt_template(provider_name, template_key)
        prompt = ChatPromptTemplate.from_template(prompt_template_str)
        log_message(f"Using prompt for refactoring TC '{tc_id}':\n{prompt_template_str}", "DEBUG")
        chain = prompt | llm | StrOutputParser()
        log_message(f"Refactor chain created for TC '{tc_id}'.", "DEBUG")

        # --- Invoke LLM ---
        status_msg = st.info(f"Asking LLM to refactor Test Case '{tc_id}'...")
        log_message(f"Invoking refactor chain for TC '{tc_id}'...", "DEBUG")
        response_str = chain.invoke({
            "tc_id": tc_id,
            "original_tc_json": original_json_str,
            "user_instructions": instructions
        })
        log_message(f"Refactor chain invocation complete for TC '{tc_id}'.", "DEBUG")
        log_message(f"Raw LLM output for refactor:\n---\n{response_str}\n---", "DEBUG")
        status_msg.empty() # Clear the "Asking LLM..." message

        if not response_str or not response_str.strip():
             log_message(f"LLM returned empty response for refactoring TC '{tc_id}'.", "WARNING")
             st.warning(f"LLM returned an empty response for refactoring TC '{tc_id}'. No changes applied.")
             return None


        # --- Parse the response as JSON object ---
        log_message(f"Attempting to parse refactor response: '{response_str[:200]}...'", "DEBUG")
        # Use the robust parser, expecting a dictionary
        updated_tc_data = parse_json_output(response_str, expected_type=dict)

        if updated_tc_data is None:
            # parse_json_output already showed error and logged
            log_message(f"Refactor failed for TC '{tc_id}': Failed to parse JSON object response.", "ERROR")
            # Add context for the user
            st.error(f"Refactoring failed for TC '{tc_id}': LLM response was not a valid JSON object.")
            return None

        # --- Validate and Process Parsed Data (updated_tc_data is a dict) ---
        # Basic validation: Check for essential keys expected in a test case
        # Make this check more flexible or configurable if needed
        essential_keys = ["Test Case ID", "Test Steps", "Test Scenario"] # Example keys
        missing_keys = [key for key in essential_keys if key not in updated_tc_data]
        if missing_keys:
            log_message(f"Refactored TC '{tc_id}' missing essential keys: {missing_keys}.", "WARNING")
            st.warning(f"Refactored TC '{tc_id}' might be incomplete (missing: {', '.join(missing_keys)}). Review result carefully.")
            # Decide whether to proceed or return None based on severity

        # Check if the Test Case ID was changed unintentionally
        new_tc_id = updated_tc_data.get("Test Case ID")
        # Compare carefully (e.g., string vs int)
        if str(new_tc_id) != str(tc_id):
            log_message(f"LLM changed TC ID during refactor from '{tc_id}' to '{new_tc_id}'.", "WARNING")
            st.warning(f"LLM changed the Test Case ID from '{tc_id}' to '{new_tc_id}'. You might need to adjust this manually.")
            # Optionally, force the ID back? Depends on requirements.
            # updated_tc_data["Test Case ID"] = tc_id

        log_message(f"Refactor successful for TC '{tc_id}'.", "INFO")
        st.success(f"Test Case '{tc_id}' refactored successfully.") # Add success message
        return updated_tc_data

    except Exception as e:
        # Catch-all for unexpected errors during refactoring
        log_message(f"Exception during refactoring for TC '{tc_id}': {type(e).__name__} - {e}", "ERROR")
        st.error(f"An unexpected error occurred during the refactoring process for TC '{tc_id}': {e}")
        return None