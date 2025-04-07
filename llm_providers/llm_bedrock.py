# Handles initialization for the AWS Bedrock provider.

import streamlit as st
import inspect # Used for checking function signatures safely
from typing import Dict, Tuple, Optional

# Langchain Core Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# AWS specific imports
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
except ImportError:
    boto3 = None # Handle missing boto3 gracefully
    NoCredentialsError, ClientError = Exception, Exception # Fallback exceptions

# Import config and utilities (adjust path if needed)
try:
    from config import DEFAULT_TEMPERATURE
    # Assuming utils.py is in the same directory or accessible via PYTHONPATH
    from helper.utils import import_class, log_message
except ImportError as e:
    try: log_message(f"CRITICAL: Failed to import required modules (config, utils) in llm_bedrock.py: {e}", "ERROR")
    except NameError: print(f"CRITICAL: Failed to import required modules (config, utils) in llm_bedrock.py: {e}")
    st.error(f"CRITICAL: Failed to import required modules (config, utils) in llm_bedrock.py: {e}")
    # Define fallbacks or stop if critical
    DEFAULT_TEMPERATURE = 0.7
    def log_message(msg, level): print(f"{level}: {msg}")
    def import_class(mod, cls): return None
    # st.stop()


def _initialize_bedrock(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """
    Initializes AWS Bedrock LLM and Embeddings based on provided configuration and credentials.
    Handles both standard and temporary (session token) AWS credentials.

    Args:
        config_dict: Configuration dictionary specific to the Bedrock provider
                     (e.g., from LLM_PROVIDER_CONFIG).
        credentials: Dictionary containing AWS credentials:
                     'aws_access_key_id', 'aws_secret_access_key', 'region_name',
                     'embedding_model_id', and optionally 'aws_session_token'.
        model_name: The specific Bedrock model ID to initialize (e.g., "anthropic.claude-v2").

    Returns:
        A tuple containing the initialized LLM and Embeddings objects, or (None, None) on failure.
    """
    log_message("Initializing AWS Bedrock provider...", "INFO")

    # --- Retrieve Credentials ---
    aws_access_key_id = credentials.get("aws_access_key_id")
    aws_secret_access_key = credentials.get("aws_secret_access_key")
    aws_session_token = credentials.get("aws_session_token") # Optional session token
    region_name = credentials.get("region_name")
    # Retrieve embedding model ID from credentials (as defined in original code)
    embedding_model_id = credentials.get("embedding_model_id")

    # --- Basic Credential Presence Check ---
    if not all([aws_access_key_id, aws_secret_access_key, region_name, embedding_model_id]):
        missing = [k for k, v in {
            "AWS Access Key ID": aws_access_key_id,
            "AWS Secret Access Key": aws_secret_access_key,
            "AWS Region Name": region_name,
            "Bedrock Embedding Model ID": embedding_model_id
        }.items() if not v]
        log_message(f"Bedrock init failed: Missing required base credentials: {', '.join(missing)}.", "ERROR")
        st.error(f"AWS Bedrock Error: Missing required credentials: {', '.join(missing)}.")
        return None, None

    # --- Session Token Check for Temporary Credentials ---
    # If the key ID looks like temporary credentials (starts with ASIA), a session token is mandatory.
    if aws_access_key_id.startswith("ASIA") and not aws_session_token:
        log_message("Bedrock init failed: Temporary credentials (Key ID starts with ASIA) require a Session Token, but it's missing.", "ERROR")
        st.error("AWS Bedrock Error: Using temporary credentials (Access Key ID starts with ASIA), but the AWS Session Token is missing in your configuration.")
        return None, None

    # --- Boto3 Installation Check ---
    if boto3 is None:
        log_message("Bedrock init failed: boto3 library not installed.", "ERROR")
        st.error("AWS Bedrock integration requires the `boto3` library. Please install it (`pip install boto3`).")
        return None, None

    # --- LangChain Class Import Check ---
    LLMClass = import_class(config_dict.get("llm_module"), config_dict.get("llm_class"))
    EmbeddingsClass = import_class(config_dict.get("embeddings_module"), config_dict.get("embeddings_class"))
    if not LLMClass or not EmbeddingsClass:
        log_message("Bedrock init failed: Could not import required LangChain classes.", "ERROR")
        st.error("Failed to import necessary LangChain classes for Bedrock. Ensure `langchain-aws` or `langchain-community` (with boto3) is installed.")
        return None, None

    try:
        # --- Prepare Boto3 Client Arguments ---
        # Create a dictionary to hold arguments for boto3.client
        boto3_client_args = {
            'service_name': 'bedrock-runtime', # Use bedrock-runtime for invoke operations
            'region_name': region_name,
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key
        }
        # Conditionally add the session token ONLY if it was provided
        if aws_session_token:
            boto3_client_args['aws_session_token'] = aws_session_token
            log_message("Using AWS Session Token for Bedrock client.", "DEBUG")
        else:
            log_message("No AWS Session Token provided; assuming permanent credentials or environment variables.", "DEBUG")

        # --- Create Boto3 Client ---
        # Use dictionary unpacking (**) to pass the arguments
        bedrock_client = boto3.client(**boto3_client_args)
        log_message(f"Boto3 client created for Bedrock Runtime in region {region_name}.", "DEBUG")

        # --- Initialize LangChain LLM ---
        # Prepare parameters for the LangChain Bedrock LLM class
        llm_params = {"client": bedrock_client, "model_id": model_name}

        # Handle temperature setting carefully (check if model_kwargs or direct temperature is expected)
        try:
            # Attempt to use model_kwargs (common in newer LangChain versions)
            llm_params["model_kwargs"] = {"temperature": DEFAULT_TEMPERATURE}
            llm = LLMClass(**llm_params)
            log_message("Passing temperature via 'model_kwargs'.", "DEBUG")
        except TypeError:
            # If model_kwargs fails, try passing temperature directly (might be older LangChain)
            del llm_params["model_kwargs"] # Remove the incorrect parameter
            sig = inspect.signature(LLMClass)
            if 'temperature' in sig.parameters:
                llm_params["temperature"] = DEFAULT_TEMPERATURE
                log_message("Passing temperature directly to LLM class.", "DEBUG")
                llm = LLMClass(**llm_params)
            else:
                # If neither works, initialize without explicit temperature (use class default)
                log_message("LLM class does not accept 'temperature' directly or via 'model_kwargs'. Using default.", "WARNING")
                llm = LLMClass(**llm_params)

        log_message(f"Bedrock LLM Class {LLMClass.__name__} initialized (model: {model_name}).", "DEBUG")

        # --- Initialize LangChain Embeddings ---
        # Ensure the embedding model ID is provided
        if not embedding_model_id:
             log_message("Bedrock Embeddings init failed: Embedding Model ID missing in credentials.", "ERROR")
             st.error("AWS Bedrock Error: Embedding Model ID is missing in the configuration.")
             return None, None # Fail initialization if embedding model is missing

        embeddings = EmbeddingsClass(client=bedrock_client, model_id=embedding_model_id)
        log_message(f"Bedrock Embeddings Class {EmbeddingsClass.__name__} initialized (model: {embedding_model_id}).", "DEBUG")

        log_message("AWS Bedrock provider initialized successfully.", "INFO")
        return llm, embeddings

    # --- Exception Handling ---
    except NoCredentialsError as e:
        log_message(f"Bedrock credentials error: {e}", "ERROR")
        st.error("AWS Bedrock Error: Credentials not found or invalid. Check configuration, environment variables (AWS_ACCESS_KEY_ID, etc.), or IAM role.")
        return None, None
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        error_msg_detail = e.response.get('Error', {}).get('Message', str(e))
        log_message(f"Bedrock client error: {error_code} - {error_msg_detail}", "ERROR")

        if error_code == 'AccessDeniedException':
            # Check if it's the specific token error or a general access denied
            if "security token" in error_msg_detail.lower() or "InvalidClientTokenId" in str(e):
                 st.error(f"AWS Bedrock Access Denied: The security token (Session Token) is invalid, expired, or does not match the access key. Please refresh your temporary credentials.")
            elif "is not authorized to perform" in error_msg_detail:
                 st.error(f"AWS Bedrock Access Denied: Check IAM permissions for 'bedrock:InvokeModel' on resource '{model_name}' and '{embedding_model_id}' in region '{region_name}'. Also ensure model access is enabled in the AWS Bedrock console.")
            else:
                 st.error(f"AWS Bedrock Access Denied: {error_msg_detail}. Check IAM permissions and Bedrock model access settings in the console.")
        elif error_code == 'ValidationException':
            st.error(f"AWS Bedrock Validation Error: Check region ('{region_name}'), model ID ('{model_name}' / '{embedding_model_id}'), or request parameters. Details: {error_msg_detail}")
        elif error_code == 'ResourceNotFoundException':
            st.error(f"AWS Bedrock Resource Not Found: Ensure model '{model_name}' or embedding model '{embedding_model_id}' is available and access is enabled in region '{region_name}'.")
        elif error_code == 'ThrottlingException':
             st.error(f"AWS Bedrock Throttling Exception: API request limit exceeded. Please wait and retry, or check your provisioned throughput if applicable. Details: {error_msg_detail}")
        elif error_code == 'ModelNotReadyException':
             st.error(f"AWS Bedrock Model Not Ready: The requested model '{model_name}' or '{embedding_model_id}' is currently unavailable. Please try again later. Details: {error_msg_detail}")
        else:
            # General ClientError
            st.error(f"AWS Bedrock ClientError: {error_code} - {error_msg_detail}")
        return None, None
    except Exception as e:
        # Catch any other unexpected errors during initialization
        error_msg = f"Unexpected error initializing AWS Bedrock: {e}"
        log_message(error_msg, "ERROR")
        st.error(error_msg)
        return None, None

    finally:
        log_message("AWS Bedrock initialization process completed.", "DEBUG")
# --- END MODIFIED: _initialize_bedrock ---