# llm_providers/llm_bedrock.py
# Handles initialization for the AWS Bedrock provider, adapted for backend.

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
    boto3 = None 
    NoCredentialsError = type('NoCredentialsError', (Exception,), {})
    ClientError = type('ClientError', (Exception,), {})

# Import config and utilities
try:
    from config import DEFAULT_TEMPERATURE
    from helper.utils import import_class, log_message
    log_message("Successfully imported config and utils in llm_bedrock.", "DEBUG")
except ImportError as e:
    print(f"LLM_BEDROCK_PROVIDER: CRITICAL: Failed to import from config or helper.utils: {e}")
    DEFAULT_TEMPERATURE = 0.7
    def log_message(msg, level, **kwargs): print(f"LLM_BEDROCK_FALLBACK_LOGGER [{level}] {msg}")
    def import_class(mod, cls): return None
    log_message("Using fallback log_message/import_class in llm_bedrock due to import error.", "ERROR")


def _initialize_bedrock(config_dict: Dict, credentials: Dict, model_name: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    """
    Initializes AWS Bedrock LLM and Embeddings.
    """
    log_message("Initializing AWS Bedrock provider...", "INFO")

    aws_access_key_id = credentials.get("aws_access_key_id")
    aws_secret_access_key = credentials.get("aws_secret_access_key")
    aws_session_token = credentials.get("aws_session_token")
    region_name = credentials.get("region_name")
    embedding_model_id = credentials.get("embedding_model_id")

    required_base_creds = {
        "AWS Access Key ID": aws_access_key_id,
        "AWS Secret Access Key": aws_secret_access_key,
        "AWS Region Name": region_name,
        "Bedrock Embedding Model ID": embedding_model_id
    }
    missing_creds = [k for k, v in required_base_creds.items() if not v]
    if missing_creds:
        log_message(f"Bedrock init failed: Missing required base credentials: {', '.join(missing_creds)}.", "ERROR")
        return None, None

    if aws_access_key_id and aws_access_key_id.startswith("ASIA") and not aws_session_token:
        log_message("Bedrock init failed: Temp creds (ASIA key ID) require Session Token, but it's missing.", "ERROR")
        return None, None

    if boto3 is None:
        log_message("Bedrock init failed: boto3 library not installed. `pip install boto3`", "ERROR")
        return None, None

    llm_module_name = config_dict.get("llm_module")
    llm_class_name = config_dict.get("llm_class")
    embeddings_module_name = config_dict.get("embeddings_module")
    embeddings_class_name = config_dict.get("embeddings_class")

    if not all([llm_module_name, llm_class_name, embeddings_module_name, embeddings_class_name]):
        log_message("Bedrock init failed: Module/class names missing in config_dict.", "ERROR")
        return None, None

    LLMClass = import_class(llm_module_name, llm_class_name)
    EmbeddingsClass = import_class(embeddings_module_name, embeddings_class_name)

    if not LLMClass or not EmbeddingsClass:
        log_message("Bedrock init failed: Could not import LangChain classes. Ensure `langchain-aws` or relevant community package is installed.", "ERROR")
        return None, None

    try:
        boto3_client_args = {
            'service_name': 'bedrock-runtime',
            'region_name': region_name,
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key
        }
        if aws_session_token:
            boto3_client_args['aws_session_token'] = aws_session_token
            log_message("Using AWS Session Token for Bedrock client.", "DEBUG")
        
        bedrock_client = boto3.client(**boto3_client_args)
        log_message(f"Boto3 client created for Bedrock Runtime in region {region_name}.", "DEBUG")

        llm_params = {"client": bedrock_client, "model_id": model_name}
        try:
            llm_params["model_kwargs"] = {"temperature": DEFAULT_TEMPERATURE}
            llm = LLMClass(**llm_params)
        except TypeError:
            del llm_params["model_kwargs"]
            sig = inspect.signature(LLMClass)
            if 'temperature' in sig.parameters:
                llm_params["temperature"] = DEFAULT_TEMPERATURE
                llm = LLMClass(**llm_params)
            else:
                log_message("LLM class does not accept 'temperature'. Using default.", "WARNING")
                llm = LLMClass(**llm_params)
        log_message(f"Bedrock LLM Class {LLMClass.__name__} initialized (model: {model_name}).", "DEBUG")

        if not embedding_model_id:
             log_message("Bedrock Embeddings init failed: Embedding Model ID missing.", "ERROR")
             return None, None
        embeddings = EmbeddingsClass(client=bedrock_client, model_id=embedding_model_id)
        log_message(f"Bedrock Embeddings Class {EmbeddingsClass.__name__} initialized (model: {embedding_model_id}).", "DEBUG")

        log_message("AWS Bedrock provider initialized successfully.", "INFO")
        return llm, embeddings

    except NoCredentialsError as e:
        log_message(f"Bedrock credentials error: {e}. Check config, env vars, or IAM role.", "ERROR")
        return None, None
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        error_msg_detail = e.response.get('Error', {}).get('Message', str(e))
        log_message(f"Bedrock client error: {error_code} - {error_msg_detail}", "ERROR", exc_info=True)
        # Specific error messages can be logged here based on error_code
        return None, None
    except Exception as e:
        log_message(f"Unexpected error initializing AWS Bedrock: {e}", "ERROR", exc_info=True)
        return None, None
    finally:
        log_message("AWS Bedrock initialization process completed.", "DEBUG")
