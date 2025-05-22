# helper/utils.py
"""General utility functions for the application, adapted for a backend environment."""

import re
import os
import importlib
import datetime
import json
import ast
import logging

# Configure basic logging for the backend
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

# Global config placeholders (normally loaded from config.py)
NO_CONTEXT_OPTION = "None"
APP_CONTEXT_FOLDER_PATH = os.path.join(os.getcwd(), "app_context") # Default, might be overridden

try:
    from config import NO_CONTEXT_OPTION as cfg_no_ctx, APP_CONTEXT_FOLDER_PATH as cfg_app_ctx_path
    NO_CONTEXT_OPTION = cfg_no_ctx
    APP_CONTEXT_FOLDER_PATH = cfg_app_ctx_path
    logger.info("Successfully imported config values into helper.utils.")
except ImportError as e:
    logger.error(f"Failed to import NO_CONTEXT_OPTION, APP_CONTEXT_FOLDER_PATH from config in helper.utils: {e}. Using defaults.")
except Exception as e:
    logger.error(f"An unexpected error occurred importing config in helper.utils: {e}. Using defaults.")


def log_message(message: str, level: str = "INFO", exc_info=False):
    """Logs a message using Python's logging module."""
    level = level.upper()
    if level == "DEBUG":
        logger.debug(message, exc_info=exc_info)
    elif level == "INFO":
        logger.info(message, exc_info=exc_info)
    elif level == "WARNING":
        logger.warning(message, exc_info=exc_info)
    elif level == "ERROR":
        logger.error(message, exc_info=exc_info)
    elif level == "CRITICAL":
        logger.critical(message, exc_info=exc_info)
    else:
        logger.info(message, exc_info=exc_info)


def import_class(module_name: str, class_name: str):
    """Dynamically imports a class from a given module."""
    log_message(f"Attempting to import {class_name} from {module_name}", "DEBUG")
    if not module_name or not class_name:
        log_message(f"Import failed: Module name ('{module_name}') or class name ('{class_name}') is empty.", "ERROR")
        return None
    try:
        module = importlib.import_module(module_name)
        imported_class = getattr(module, class_name)
        log_message(f"Successfully imported {class_name} from {module_name}", "DEBUG")
        return imported_class
    except ImportError:
        package_name = module_name.replace('_', '-')
        if package_name == 'langchain-google-genai': package_name = 'langchain-google-genai'
        elif package_name == 'langchain-aws': package_name = 'langchain-aws boto3'
        elif package_name == 'langchain-anthropic': package_name = 'langchain-anthropic'
        elif package_name == 'langchain-groq': package_name = 'langchain-groq'
        log_message(f"Failed to import `{class_name}` from `{module_name}`. Install: `pip install {package_name}`", "ERROR")
        return None
    except AttributeError:
        log_message(f"Class `{class_name}` not found in module `{module_name}`.", "ERROR")
        return None
    except Exception as e:
        log_message(f"Unexpected error during import of {class_name} from {module_name}: {e}", "ERROR", exc_info=True)
        return None


def sanitize_filename(name: str, max_length: int = 200) -> str:
    """Removes invalid characters for filenames and optionally truncates."""
    if not isinstance(name, str):
        name = str(name)
    name = name.replace(" ", "_")
    name = re.sub(r'[<>:"/\\|?*\[\]\']', '', name)
    name = name.strip('._ ')
    return name[:max_length]


def get_available_context_files() -> list[str]:
    """
    Scans the configured context folder for .yaml/.yml files.
    Note: This function's relevance might change as context handling evolves.
    """
    context_files = [NO_CONTEXT_OPTION]
    folder = APP_CONTEXT_FOLDER_PATH
    log_message(f"Scanning for context files (legacy method) in: {folder}", "DEBUG")
    if os.path.exists(folder) and os.path.isdir(folder):
        try:
            for filename in os.listdir(folder):
                if filename.lower().endswith((".yaml", ".yml")):
                    base_name = os.path.splitext(filename)[0]
                    context_files.append(base_name)
                    log_message(f"Found context file (legacy scan): {filename}", "DEBUG")
        except OSError as e:
            log_message(f"OS Error scanning context folder '{folder}': {e}", "WARNING", exc_info=True)
        except Exception as e:
            log_message(f"Unexpected error scanning context folder '{folder}': {e}", "WARNING", exc_info=True)
    else:
        log_message(f"Context folder not found (legacy scan): {folder}", "INFO")
    return sorted(list(set(context_files)))


def parse_json_output(llm_output: str, expected_type: type = list):
    """
    Attempts to parse JSON or Python literal lists/dicts from LLM output,
    handling markdown code blocks and preferring JSON but falling back to ast.literal_eval.
    """
    log_message(f"Attempting to parse LLM output as {expected_type.__name__}", "DEBUG")
    if not llm_output:
        log_message("Parsing failed: LLM output is empty.", "WARNING")
        return None

    extracted_str = None
    json_block_match = re.search(r'```json\s*(.*)\s*```', llm_output, re.DOTALL | re.IGNORECASE)
    if json_block_match:
        log_message("Found ```json block.", "DEBUG")
        extracted_str = json_block_match.group(1).strip()
    else:
        code_block_match = re.search(r'```\s*(.*)\s*```', llm_output, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            log_message("Found generic ``` block.", "DEBUG")
            extracted_str = code_block_match.group(1).strip()
        else:
            log_message("No code blocks found, searching for raw list/dict.", "DEBUG")
            raw_pattern = r'(\[.*\])' if expected_type == list else r'(\{.*\})'
            raw_match = re.search(raw_pattern, llm_output, re.DOTALL)
            if raw_match:
                 log_message("Found raw list/dict pattern match.", "DEBUG")
                 extracted_str = raw_match.group(1).strip()
            else:
                 log_message("No raw pattern match, checking full trimmed string.", "DEBUG")
                 trimmed_output = llm_output.strip()
                 looks_like_list = expected_type is list and trimmed_output.startswith('[') and trimmed_output.endswith(']')
                 looks_like_dict = expected_type is dict and trimmed_output.startswith('{') and trimmed_output.endswith('}')
                 if looks_like_list or looks_like_dict:
                      extracted_str = trimmed_output
                      log_message("Using full trimmed output as potential literal.", "DEBUG")

    if not extracted_str:
         log_message(f"Could not extract a plausible {expected_type.__name__} structure from LLM output.", "WARNING")
         return None

    parsed_data = None
    try:
        log_message(f"Attempting json.loads on extracted string: '{extracted_str[:100]}...'", "DEBUG")
        parsed_data = json.loads(extracted_str)
        log_message("json.loads successful.", "INFO")
    except json.JSONDecodeError as json_err:
        log_message(f"json.loads failed ({json_err}). Trying ast.literal_eval.", "WARNING")
        try:
            log_message(f"Attempting ast.literal_eval on extracted string: '{extracted_str[:100]}...'", "DEBUG")
            parsed_data = ast.literal_eval(extracted_str)
            log_message("ast.literal_eval successful.", "INFO")
        except (ValueError, SyntaxError, MemoryError, TypeError) as ast_err:
            log_message(f"ast.literal_eval also failed: {type(ast_err).__name__} - {ast_err}", "ERROR")
            return None
        except Exception as e:
            log_message(f"Unexpected error during ast.literal_eval: {type(e).__name__} - {e}", "ERROR", exc_info=True)
            return None

    if parsed_data is not None:
        if isinstance(parsed_data, expected_type):
            log_message(f"Parsed data type matches expected type ({expected_type.__name__}).", "INFO")
            return parsed_data
        else:
            log_message(f"Parsed data type is {type(parsed_data).__name__}, expected {expected_type.__name__}.", "WARNING")
            return None

    log_message("Parsing failed through unknown means (e.g. extracted_str was valid but then parsing failed without exception).", "ERROR")
    return None
