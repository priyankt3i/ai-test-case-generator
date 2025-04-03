# llm_integration.py
"""Handles LangChain setup, LLM interactions (identification, RAG, refactoring)."""

import streamlit as st
import yaml
import os
import json
from typing import Dict, Any, Tuple, List, Optional
import requests

# Langchain Core Imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain # Using original method
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
# *** ADD OLLAMA Imports ***
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings

# Import config and utilities
try:
    from config import (
        LLM_PROVIDER_CONFIG, FALLBACK_EMBEDDING_PROVIDERS,
        DEFAULT_TEMPERATURE, APP_CONTEXT_FOLDER_PATH, NO_CONTEXT_OPTION,
        CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_SEARCH_K,
        IDENTIFY_APP_PROMPT_TEMPLATE, GENERATE_TC_PROMPT_TEMPLATE,
        REFACTOR_TC_PROMPT_TEMPLATE, EXCEL_EXPECTED_COLUMNS
    )
    # *** Import the logger ***
    from utils import import_class, parse_json_output, log_message
except ImportError as e:
    # Log critical import errors if possible, otherwise use st.error
    try:
        log_message(f"CRITICAL: Failed to import required modules (config, utils). Ensure they exist: {e}", "ERROR")
    except NameError: # log_message itself failed to import
        pass
    st.error(f"CRITICAL: Failed to import required modules (config, utils). Ensure they exist: {e}")
    st.stop() # Stop app if core components missing


# Specific exception imports (add more as needed)
try: from botocore.exceptions import NoCredentialsError, ClientError
except ImportError: NoCredentialsError, ClientError = Exception, Exception
try: from openai import AuthenticationError as OpenAIAuthenticationError, RateLimitError as OpenAIRateLimitError
except ImportError: OpenAIAuthenticationError, OpenAIRateLimitError = Exception, Exception
try: from google.api_core.exceptions import PermissionDenied as GooglePermissionDenied, ResourceExhausted as GoogleResourceExhausted
except ImportError: GooglePermissionDenied, GoogleResourceExhausted = Exception, Exception
# Anthropic/Groq might have their own specific exceptions, import if available/needed

# --- Provider Initialization Helpers ---
# NOTE: Provider initialization functions (_initialize_openai, etc.) remain unchanged.
# Added logging within them.
# (Code omitted for brevity - see previous versions, just add log_message calls)
def _initialize_openai(config: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """Initializes OpenAI LLM and Embeddings."""
    log_message("Initializing OpenAI provider...", "INFO")
    api_key = credentials.get("api_key")
    if not api_key:
        log_message("OpenAI init failed: API key missing.", "ERROR")
        return None, None

    LLMClass = import_class(config["llm_module"], config["llm_class"])
    EmbeddingsClass = import_class(config["embeddings_module"], config["embeddings_class"])
    if not LLMClass or not EmbeddingsClass:
        log_message("OpenAI init failed: Could not import required LangChain classes.", "ERROR")
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

# (Add similar logging to _initialize_gemini, _initialize_claude, _initialize_bedrock, _initialize_groq, _get_fallback_embeddings)
# ... Example for Gemini ...
def _initialize_gemini(config: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """Initializes Gemini LLM and Embeddings."""
    log_message("Initializing Gemini provider...", "INFO")
    api_key = credentials.get("api_key")
    if not api_key:
        log_message("Gemini init failed: API key missing.", "ERROR")
        return None, None

    LLMClass = import_class(config["llm_module"], config["llm_class"])
    EmbeddingsClass = import_class(config["embeddings_module"], config["embeddings_class"])
    if not LLMClass or not EmbeddingsClass:
        log_message("Gemini init failed: Could not import required LangChain classes.", "ERROR")
        return None, None

    try:
        llm = LLMClass(google_api_key=api_key, model=model_name, temperature=DEFAULT_TEMPERATURE, convert_system_message_to_human=True)
        log_message(f"Gemini LLM Class {LLMClass.__name__} initialized.", "DEBUG")
        embed_model_id = config.get("embeddings_model_id", "models/embedding-001")
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

# ... (Assume other init helpers are similarly updated with logging) ...
def _initialize_claude(config: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """Initializes Claude LLM. Embeddings handled by fallback."""
    log_message("Initializing Claude provider...", "INFO")
    api_key = credentials.get("api_key")
    if not api_key:
        log_message("Claude init failed: API key missing.", "ERROR")
        return None, None

    LLMClass = import_class(config["llm_module"], config["llm_class"])
    if not LLMClass:
        log_message("Claude init failed: Could not import required LangChain class.", "ERROR")
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

def _initialize_bedrock(config: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """Initializes AWS Bedrock LLM and Embeddings."""
    log_message("Initializing AWS Bedrock provider...", "INFO")
    aws_access_key_id = credentials.get("aws_access_key_id")
    aws_secret_access_key = credentials.get("aws_secret_access_key")
    region_name = credentials.get("region_name")
    embedding_model_id = credentials.get("embedding_model_id")

    if not all([aws_access_key_id, aws_secret_access_key, region_name, embedding_model_id]):
        log_message("Bedrock init failed: Missing one or more credentials.", "ERROR")
        return None, None

    try: import boto3
    except ImportError:
        log_message("Bedrock init failed: boto3 not installed.", "ERROR")
        st.error("AWS Bedrock requires `boto3`. Install it (`pip install boto3`).")
        return None, None

    LLMClass = import_class(config["llm_module"], config["llm_class"])
    EmbeddingsClass = import_class(config["embeddings_module"], config["embeddings_class"])
    if not LLMClass or not EmbeddingsClass:
        log_message("Bedrock init failed: Could not import required LangChain classes.", "ERROR")
        return None, None

    try:
        bedrock_client = boto3.client(
            service_name='bedrock-runtime', region_name=region_name,
            aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key
        )
        log_message(f"Bedrock client created for region {region_name}.", "DEBUG")

        llm_params = {"client": bedrock_client, "model_id": model_name}
        if hasattr(LLMClass, "_default_params") and "temperature" in LLMClass._default_params:
             llm_params["temperature"] = DEFAULT_TEMPERATURE
        else: llm_params["model_kwargs"] = {"temperature": DEFAULT_TEMPERATURE}
        llm = LLMClass(**llm_params)
        log_message(f"Bedrock LLM Class {LLMClass.__name__} initialized (model: {model_name}).", "DEBUG")

        embeddings = EmbeddingsClass(client=bedrock_client, model_id=embedding_model_id)
        log_message(f"Bedrock Embeddings Class {EmbeddingsClass.__name__} initialized (model: {embedding_model_id}).", "DEBUG")
        log_message("AWS Bedrock provider initialized successfully.", "INFO")
        return llm, embeddings

    except NoCredentialsError as e:
        log_message(f"Bedrock credentials error: {e}", "ERROR")
        st.error("AWS Bedrock Error: Credentials not found. Check configuration.")
        return None, None
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        error_msg = e.response.get('Error', {}).get('Message', str(e))
        log_message(f"Bedrock client error: {error_code} - {error_msg}", "ERROR")
        if error_code == 'AccessDeniedException': st.error(f"AWS Bedrock Access Denied: Check IAM permissions for Bedrock and model '{model_name}' or '{embedding_model_id}' in region '{region_name}'.")
        elif error_code == 'ValidationException': st.error(f"AWS Bedrock Validation Error: Check region '{region_name}' or model ID '{model_name}' / '{embedding_model_id}'. {error_msg}")
        elif error_code == 'ResourceNotFoundException': st.error(f"AWS Bedrock Resource Not Found: Ensure model '{model_name}' or '{embedding_model_id}' is available/enabled in region '{region_name}'.")
        else: st.error(f"AWS Bedrock ClientError: {error_code} - {error_msg}")
        return None, None
    except Exception as e:
        error_msg = f"Error initializing AWS Bedrock: {e}"
        log_message(error_msg, "ERROR")
        st.error(error_msg)
        return None, None

def _initialize_groq(config: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """Initializes Groq LLM. Embeddings handled by fallback."""
    log_message("Initializing Groq provider...", "INFO")
    api_key = credentials.get("api_key")
    if not api_key:
        log_message("Groq init failed: API key missing.", "ERROR")
        return None, None

    LLMClass = import_class(config["llm_module"], config["llm_class"])
    if not LLMClass:
        log_message("Groq init failed: Could not import required LangChain class.", "ERROR")
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
        st.error("RAG requires OpenAI API key for fallback embeddings, but it's missing.")
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
    
# *** NEW OLLAMA INITIALIZATION HELPER ***
def _initialize_ollama(config: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """Initializes Ollama LLM and Embeddings."""
    log_message("Initializing Ollama provider...", "INFO")
    # Provide default URL if not entered by user
    base_url = credentials.get("base_url", "http://localhost:11434").strip()
    if not base_url: # Handle empty string case
        base_url = "http://localhost:11434"
        log_message("Ollama Base URL empty, using default: http://localhost:11434", "WARNING")
    # Model name comes from the dropdown selection, passed as model_name argument
    if not model_name:
        log_message("Ollama init failed: Model name missing.", "ERROR")
        st.error("Ollama requires a model to be selected.")
        return None, None
    log_message(f"Ollama using Base URL: {base_url}, Model: {model_name}", "DEBUG")
    # Check if Ollama server is reachable before initializing LangChain components
    try:
        response = requests.get(base_url, timeout=5) # Check base endpoint
        response.raise_for_status() # Raise exception for bad status codes
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
    except requests.exceptions.RequestException as e:
        error_msg = f"Ollama Request Error: Failed to query {base_url}. Error: {e}"
        log_message(error_msg, "ERROR")
        st.error(error_msg)
        return None, None
    # Proceed with LangChain initialization if connection test passed
    try:
        # Note: LangChain's Ollama classes handle module import internally if needed
        llm = ChatOllama(
            base_url=base_url,
            model=model_name,
            temperature=DEFAULT_TEMPERATURE
        )
        log_message(f"Ollama LLM Class ChatOllama initialized.", "DEBUG")
        # Using the same model name for embeddings by default.
        # For production, consider allowing a separate embedding model config.
        embeddings = OllamaEmbeddings(
            base_url=base_url,
            model=model_name
        )
        log_message(f"Ollama Embeddings Class OllamaEmbeddings initialized (using model: {model_name}).", "DEBUG")
        # Optional: Add a quick test call to ensure model is available
        try:
             llm.invoke("Hi")
             log_message(f"Ollama model '{model_name}' responded successfully.", "DEBUG")
        except Exception as model_e:
             # Catch errors specifically related to the model (e.g., not found)
             error_msg = f"Ollama Model Error: Failed to invoke model '{model_name}'. Is it pulled? Error: {model_e}"
             log_message(error_msg, "ERROR")
             st.error(error_msg)
             return None, None # Fail initialization if model invocation fails
        log_message("Ollama provider initialized successfully.", "INFO")
        return llm, embeddings
    except Exception as e:
        error_msg = f"Error initializing Ollama LangChain components: {e}"
        log_message(error_msg, "ERROR")
        st.error(error_msg)
        return None, None
# *** END NEW OLLAMA HELPER ***

# --- Main Initialization Function ---
# (Code omitted for brevity)
def get_llm_and_embeddings(provider: str, model_name: str, credentials: Dict, fallback_openai_key: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """Initializes LangChain LLM and Embeddings objects based on the selected provider."""
    log_message(f"Getting LLM and Embeddings for provider: {provider}, model: {model_name}", "INFO")
    llm: Optional[BaseChatModel] = None
    embeddings: Optional[Embeddings] = None
    config = LLM_PROVIDER_CONFIG.get(provider)

    if not config:
        log_message(f"Invalid provider selected: {provider}", "ERROR")
        st.error(f"Invalid provider selected: {provider}")
        return None, None

    is_fallback_provider = provider in FALLBACK_EMBEDDING_PROVIDERS
    log_message(f"Is fallback provider? {is_fallback_provider}", "DEBUG")

    init_functions = {
        "OpenAI": _initialize_openai, "Gemini": _initialize_gemini,
        "Claude": _initialize_claude, "AWS Bedrock": _initialize_bedrock,
        "Groq": _initialize_groq,
        "Ollama": _initialize_ollama, # Added Ollama
    }
    init_func = init_functions.get(provider)
    if not init_func:
        log_message(f"Initialization logic not defined for provider: {provider}", "ERROR")
        st.error(f"Initialization logic not defined for provider: {provider}")
        return None, None

    llm, embeddings = init_func(config, credentials, model_name)

    if is_fallback_provider and not embeddings:
        log_message(f"Provider {provider} requires fallback embeddings.", "INFO")
        embeddings = _get_fallback_embeddings(fallback_openai_key)
        if not embeddings:
            log_message(f"Failed to initialize fallback embeddings for {provider}.", "WARNING")
            st.warning(f"RAG may fail for {provider} as fallback embeddings could not be initialized.")

    if not llm:
        log_message(f"LLM initialization final check failed for {provider}.", "ERROR")
        # Error message already shown by helper
        if not is_fallback_provider or not embeddings: return None, None

    if not embeddings:
        if provider == "Ollama":
            log_message(f"Ollama embeddings failed to initialize even if LLM succeeded.", "ERROR")
            st.error(f"Ollama embeddings failed. RAG will not work.")
        elif not is_fallback_provider: # Non-fallback, non-Ollama providers
            log_message(f"Embeddings initialization final check failed for {provider}. RAG will not work.", "ERROR")
            st.error(f"Embeddings initialization failed for {provider}. RAG features will not work.")
        # Allow fallback providers to proceed without embeddings if LLM is ok
        return llm, None # Return LLM if it succeeded

    log_message(f"LLM and Embeddings initialized successfully for {provider}.", "INFO")
    return llm, embeddings

# --- Credential Checking ---
# (Code omitted for brevity)
def check_credentials(provider: str, credentials: Dict, fallback_key: str, require_fallback_for_rag: bool) -> Tuple[bool, str]:
    """Validates if all necessary credentials are present."""
    # ... (logic remains the same) ...
    config = LLM_PROVIDER_CONFIG.get(provider)
    if not config: return False, "Invalid provider selected."
    missing = []
    required_creds = config.get("credentials", [])
    for key in required_creds:
        # For Ollama, 'model' comes from dropdown, 'base_url' might be empty (use default)
        if provider == "Ollama" and key == "base_url":
             continue # Don't require base_url to be non-empty, use default
        if provider == "Ollama" and key == "model":
             # Model is selected via dropdown, not direct credential input usually
             # Check if model_name exists in session state instead? Or rely on later check.
             # Let's skip strict check here, rely on get_llm_and_embeddings check.
             continue
        # Standard check for other providers/credentials
        if not credentials.get(key, "").strip():
            missing.append(key.replace("_", " ").title())
        if not credentials.get(key, "").strip(): missing.append(key.replace("_", " ").title())
    # Check for fallback key requirement
    needs_fallback = provider in FALLBACK_EMBEDDING_PROVIDERS
    if require_fallback_for_rag and needs_fallback and not fallback_key.strip(): missing.append("OpenAI API Key (for RAG Fallback)")
    if missing: return False, f"Missing credentials for {provider}: {', '.join(missing)}."
    return True, ""


# --- LLM Interaction Functions ---
# (Code omitted for brevity)
def identify_applications(text: str, llm: BaseChatModel) -> List[str]:
    """Identifies application names from text using the provided LLM."""
    log_message("Starting application identification...", "INFO")
    if not text:
        log_message("Identification failed: Input text is empty.", "ERROR")
        st.error("Cannot identify applications: Input text is empty.")
        return []
    if not llm:
         log_message("Identification failed: LLM is not initialized.", "ERROR")
         st.error("Cannot identify applications: LLM is not initialized.")
         return []

    try:
        app_prompt = ChatPromptTemplate.from_template(IDENTIFY_APP_PROMPT_TEMPLATE)
        app_chain = app_prompt | llm | StrOutputParser()
        log_message("Application identification chain created.", "DEBUG")

        with st.spinner(f"Asking LLM ({llm.__class__.__name__}) to identify applications..."):
             result_str = app_chain.invoke({"text": text})
             log_message("LLM invocation for identification complete.", "DEBUG")

        parsed_apps = parse_json_output(result_str, expected_type=list)

        if parsed_apps is None:
            log_message("Failed to parse application list from LLM output.", "WARNING")
            if result_str and not result_str.startswith("I cannot") and len(result_str) < 200 and '[' not in result_str and '{' not in result_str:
                 possible_apps = [app.strip().strip("'\"") for app in result_str.split(',') if app.strip()]
                 if possible_apps:
                     log_message("Attempting basic comma parsing for app list.", "INFO")
                     st.info("LLM output wasn't a JSON list, attempting basic comma parsing.")
                     return sorted(list(set(possible_apps)))
            return []

        cleaned_apps = [str(app).strip() for app in parsed_apps if str(app).strip()]
        log_message(f"Identification successful. Found apps: {cleaned_apps}", "INFO")
        return sorted(list(set(cleaned_apps)))

    except Exception as e:
        log_message(f"Exception during application identification: {type(e).__name__} - {e}", "ERROR")
        st.error(f"An error occurred during application identification LLM call: {e}")
        return []


# *** generate_test_cases FUNCTION WITH ENHANCED DEBUGGING LOGGING ***
def generate_test_cases(
    text: str,
    selected_apps: List[str],
    context_selections: Dict[str, str],
    llm: BaseChatModel,
    embeddings: Embeddings
) -> Dict[str, Any]:
    """
    Generates test cases for selected applications using RAG.
    Uses the create_retrieval_chain method based on original working code.
    Includes enhanced debugging logging.

    Args:
        text: The original requirements text.
        selected_apps: List of application names to generate cases for.
        context_selections: Dict mapping app names to selected context file base names.
        llm: Initialized LangChain Chat Model.
        embeddings: Initialized LangChain Embeddings model.

    Returns:
        A dictionary where keys are app names and values are either a list
        of generated test case dicts or an error string.
    """
    log_message("--- Entered generate_test_cases ---", "DEBUG") # Use DEBUG level

    results = {}
    if not selected_apps:
        log_message("Generation skipped: No applications selected.", "WARNING")
        st.warning("No applications selected for test case generation.")
        return results
    if not text:
        log_message("Generation failed: Extracted text is empty.", "ERROR")
        st.error("Cannot generate test cases: Extracted text is empty.")
        return {app: "Error: Source text is empty." for app in selected_apps}
    if not llm:
         log_message("Generation failed: LLM is not initialized.", "ERROR")
         st.error("Cannot generate test cases: LLM is not initialized.")
         return {app: "Error: LLM not initialized." for app in selected_apps}
    if not embeddings:
        log_message("Generation failed: Embeddings are not initialized.", "ERROR")
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
            vectorstore = FAISS.from_texts(splits, embedding=embeddings)
        # No st.success here, log instead
        log_message("--- Vector Store Creation Succeeded ---", "INFO")
    except Exception as e:
        log_message(f"--- Exception during Vector Store Creation ---", "ERROR")
        log_message(f"Caught Exception Type: {type(e).__name__}", "ERROR")
        log_message(f"Caught Exception Details: {e}", "ERROR")
        st.error(f"Error creating vector store: {e}")
        err_str = str(e).lower()
        # Keep specific user-facing errors
        if "authenticate" in err_str or "permission" in err_str or "forbidden" in err_str or "api key not valid" in err_str:
             st.error("Authentication/Permission error during embedding. Check credentials for the embedding provider.")
        elif "quota" in err_str or "limit" in err_str or "resourceexhausted" in err_str:
             st.error(f"API Quota/Rate Limit likely exceeded for embedding provider.")
        elif "model" in err_str and ("not found" in err_str or "invalid" in err_str):
             st.error(f"Embedding Model not found or invalid.")
        return {app: f"Error creating embeddings: {e}" for app in selected_apps}

    if vectorstore is None:
        log_message("Vector store object is None after creation block.", "ERROR")
        st.error("Cannot proceed: Vector store object is None after creation block.")
        return {app: "Error: Vector store initialization failed unexpectedly." for app in selected_apps}

    # 2. Setup RAG Chain
    retrieval_chain = None
    try:
        log_message("--- Entering RAG Setup Try Block ---", "DEBUG")
        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_SEARCH_K})
        log_message(f"Retriever Type: {type(retriever)}", "DEBUG")

        tc_fields = ", ".join([f"`{col}`" for col in EXCEL_EXPECTED_COLUMNS])
        prompt_template_str = GENERATE_TC_PROMPT_TEMPLATE.format(field_names=tc_fields)
        log_message("--- Attempting ChatPromptTemplate.from_template ---", "DEBUG")
        test_case_prompt = ChatPromptTemplate.from_template(prompt_template_str)
        prompt_vars = getattr(test_case_prompt, 'input_variables', 'N/A')
        log_message(f"Prompt Type: {type(test_case_prompt)}, Input Variables: {prompt_vars}", "DEBUG")

        log_message(f"LLM Type: {type(llm)}", "DEBUG")
        log_message("--- Attempting create_stuff_documents_chain ---", "DEBUG")
        document_chain = create_stuff_documents_chain(llm, test_case_prompt)
        log_message("--- create_stuff_documents_chain Succeeded ---", "DEBUG")
        doc_chain_keys = getattr(document_chain, 'input_keys', 'N/A')
        log_message(f"Document Chain Type: {type(document_chain)}, Input Keys: {doc_chain_keys}", "DEBUG")

        log_message("--- Attempting create_retrieval_chain ---", "DEBUG")
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        log_message("--- create_retrieval_chain Succeeded ---", "INFO")

    except Exception as e:
        log_message("--- Exception during RAG Setup ---", "ERROR")
        log_message(f"Caught Exception Type: {type(e).__name__}", "ERROR")
        log_message(f"Caught Exception Details: {e}", "ERROR")

        # Keep user-facing error
        st.error(f"Fatal error setting up RAG chain components: {e}")
        if "'context'" in str(e).lower():
             st.error("This indicates an issue passing retrieved documents ('context') to the prompt. Check chain setup and prompt variables based on debug output above.")

        # Populate results dict with the error for display later
        error_message_for_results = f"Error setting up RAG chain: {e}"
        for app in selected_apps:
             if app not in results:
                 results[app] = error_message_for_results
        return results # Stop execution if setup fails

    # 3. Generate for each selected application
    if not retrieval_chain:
         log_message("Cannot proceed with generation as RAG chain setup failed (retrieval_chain is None).", "ERROR")
         st.error("Cannot proceed with generation as RAG chain setup failed.")
         return results # Should already contain setup error message

    total_selected_apps = len(selected_apps)
    log_message(f"Starting generation loop for {total_selected_apps} selected apps.", "INFO")
    progress_bar = st.progress(0.0, text="Initializing test case generation...")

    for i, app_name in enumerate(selected_apps):
        log_message(f"Processing app {i+1}/{total_selected_apps}: {app_name}", "INFO")
        progress_value = (i + 1) / total_selected_apps
        progress_text = f"Generating for '{app_name}' ({i+1}/{total_selected_apps})..."
        progress_bar.progress(progress_value, text=progress_text)

        with st.status(f"Processing '{app_name}'...", expanded=False) as status:
            try:
                # --- Context File Loading ---
                yaml_context_str = ""
                selected_yaml_base = context_selections.get(app_name, NO_CONTEXT_OPTION)
                log_message(f"App '{app_name}': Selected context file base = '{selected_yaml_base}'", "DEBUG")
                status.write(f"Selected Context File: '{selected_yaml_base}'")
                if selected_yaml_base != NO_CONTEXT_OPTION:
                    # ... (YAML loading logic - add logging inside if needed) ...
                    yaml_filename = f"{selected_yaml_base}.yaml"
                    yaml_path = os.path.join(APP_CONTEXT_FOLDER_PATH, yaml_filename)
                    if not os.path.exists(yaml_path):
                         yaml_filename = f"{selected_yaml_base}.yml"
                         yaml_path = os.path.join(APP_CONTEXT_FOLDER_PATH, yaml_filename)
                    if os.path.exists(yaml_path):
                        log_message(f"App '{app_name}': Loading context from {yaml_path}", "DEBUG")
                        try:
                            with open(yaml_path, 'r', encoding='utf-8') as yf:
                                yaml_data = yaml.safe_load(yf)
                                yaml_context_str = f"\n\n--- Additional Context ({yaml_filename}) ---\n{yaml.dump(yaml_data, indent=2, allow_unicode=True)}\n--- End Context ---"
                            log_message(f"App '{app_name}': Context loaded successfully.", "DEBUG")
                            status.write(f"Successfully loaded context from {yaml_filename}")
                        except yaml.YAMLError as ye: status.warning(f"⚠️ Error parsing YAML file {yaml_filename}: {ye}"); log_message(f"App '{app_name}': YAML parse error - {ye}", "WARNING")
                        except OSError as oe: status.warning(f"⚠️ Error reading YAML file {yaml_filename}: {oe}"); log_message(f"App '{app_name}': YAML read error - {oe}", "WARNING")
                        except Exception as e: status.warning(f"⚠️ Unexpected error loading context {yaml_filename}: {e}"); log_message(f"App '{app_name}': YAML load error - {e}", "WARNING")
                    else: status.warning(f"⚠️ Context file '{selected_yaml_base}.yaml/.yml' not found in '{APP_CONTEXT_FOLDER_PATH}'."); log_message(f"App '{app_name}': Context file not found at {yaml_path}", "WARNING")
                # --- End Context File Loading ---

                input_query_string = f"Generate test cases specifically relevant to the application or system named: '{app_name}'. Use the retrieved requirements context and the additional context provided below (if any) to inform the test cases.{yaml_context_str}"
                log_message(f"App '{app_name}': Prepared input query.", "DEBUG")

                status.write("Invoking RAG chain...")
                log_message(f"App '{app_name}': Invoking retrieval_chain...", "DEBUG")
                response = retrieval_chain.invoke({"input": input_query_string})
                log_message(f"App '{app_name}': Received response from retrieval_chain.", "DEBUG")
                status.write("Received response from LLM.")

                # Process response
                if isinstance(response, dict) and "answer" in response and response["answer"]:
                    answer_str = response["answer"].strip()
                    log_message(f"App '{app_name}': Got answer from response dict. Length: {len(answer_str)}", "DEBUG")
                    parsed_cases = parse_json_output(answer_str, expected_type=list)

                    if parsed_cases is None:
                        results[app_name] = "Error: Failed to parse JSON list from LLM response."
                        log_message(f"App '{app_name}': Failed to parse JSON.", "ERROR")
                        status.update(label="⚠️ JSON Parse Error", state="error", expanded=True)
                    elif isinstance(parsed_cases, list):
                         if not parsed_cases:
                              results[app_name] = "Warning: LLM returned an empty list of test cases."
                              log_message(f"App '{app_name}': LLM returned empty list.", "WARNING")
                              status.update(label="⚠️ Empty List", state="warning")
                         else:
                              if all(isinstance(item, dict) for item in parsed_cases):
                                 results[app_name] = parsed_cases
                                 log_message(f"App '{app_name}': Successfully generated {len(parsed_cases)} cases.", "INFO")
                                 status.update(label=f"✓ Generated {len(parsed_cases)} cases", state="complete")
                              else:
                                 results[app_name] = "Error: LLM list items are not all JSON objects."
                                 log_message(f"App '{app_name}': Generated list contains non-dict items.", "ERROR")
                                 status.update(label="⚠️ Invalid Item Type", state="error", expanded=True)
                else:
                    results[app_name] = "Error: LLM provided no answer or unexpected response structure."
                    log_message(f"App '{app_name}': No 'answer' key in response or response not a dict. Response: {response}", "ERROR")
                    status.update(label="⚠️ No Answer/Bad Format", state="error", expanded=True)

            except Exception as e:
                log_message(f"--- Exception during Generation Loop for '{app_name}' ---", "ERROR")
                log_message(f"Caught Exception Type: {type(e).__name__}", "ERROR")
                log_message(f"Caught Exception Details: {e}", "ERROR")
                st.error(f"An error occurred during generation for '{app_name}': {e}")
                results[app_name] = f"Error: Generation failed - {e}"
                status.update(label="❌ Failed", state="error", expanded=True)


    progress_bar.empty()
    log_message("--- Finished generate_test_cases ---", "DEBUG")
    return results


# --- refactor_single_test_case Function ---
# (Code omitted for brevity - add logging similarly if needed)
def refactor_single_test_case(
    app_name: str,
    tc_id: str,
    instructions: str,
    original_tc_data: Dict,
    llm: BaseChatModel
) -> Optional[Dict]:
    """Uses LLM to refactor a single test case based on instructions."""
    log_message(f"Starting refactor for TC '{tc_id}' in app '{app_name}'...", "INFO")
    if not llm:
        st.error("Cannot refactor: LLM is not initialized.")
        log_message("Refactor failed: LLM not initialized.", "ERROR")
        return None
    if not original_tc_data or not isinstance(original_tc_data, dict):
        st.error(f"Cannot refactor: Invalid original test case data provided for TC ID '{tc_id}'.")
        log_message(f"Refactor failed: Invalid original data for TC '{tc_id}'.", "ERROR")
        return None
    if not instructions:
         st.warning("Cannot refactor: Modification instructions are empty.")
         log_message(f"Refactor skipped: Empty instructions for TC '{tc_id}'.", "WARNING")
         return None

    try:
        original_json_str = json.dumps(original_tc_data, indent=2)
        prompt = ChatPromptTemplate.from_template(REFACTOR_TC_PROMPT_TEMPLATE)
        chain = prompt | llm | StrOutputParser()
        log_message(f"Refactor chain created for TC '{tc_id}'.", "DEBUG")

        status_msg = st.info(f"Asking LLM to refactor Test Case '{tc_id}'...")
        log_message(f"Invoking refactor chain for TC '{tc_id}'...", "DEBUG")
        response_str = chain.invoke({
            "tc_id": tc_id, "original_tc_json": original_json_str,
            "user_instructions": instructions
        })
        log_message(f"Refactor chain invocation complete for TC '{tc_id}'.", "DEBUG")
        status_msg.empty()

        updated_tc_data = parse_json_output(response_str, expected_type=dict)

        if updated_tc_data is None:
            log_message(f"Refactor failed for TC '{tc_id}': Failed to parse JSON response.", "ERROR")
            return None

        if isinstance(updated_tc_data, dict):
            if "Test Case ID" not in updated_tc_data or "Test Steps" not in updated_tc_data:
                 log_message(f"Refactored TC '{tc_id}' missing essential keys.", "WARNING")
                 st.warning("Refactored JSON is missing essential keys ('Test Case ID', 'Test Steps'). Review result carefully.")
            new_tc_id = updated_tc_data.get("Test Case ID")
            if new_tc_id != tc_id:
                 log_message(f"LLM changed TC ID from '{tc_id}' to '{new_tc_id}'.", "WARNING")
                 st.warning(f"LLM changed the Test Case ID from '{tc_id}' to '{new_tc_id}'.")
            log_message(f"Refactor successful for TC '{tc_id}'.", "INFO")
            return updated_tc_data
        else:
             # Should be caught by parse_json_output, but as fallback
             log_message(f"Refactor failed for TC '{tc_id}': Unexpected parse result type {type(updated_tc_data).__name__}.", "ERROR")
             st.error(f"Refactoring failed: Expected a JSON object, but parsing returned type {type(updated_tc_data).__name__}.")
             return None

    except Exception as e:
        log_message(f"Exception during refactoring for TC '{tc_id}': {type(e).__name__} - {e}", "ERROR")
        st.error(f"An error occurred during the refactoring LLM call: {e}")
        return None
