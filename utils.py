# utils.py
"""General utility functions for the application."""

import re
import os
import importlib
import streamlit as st
import datetime # Added for timestamping logs
import json # Keep for parse_json_output

# Make sure config.py is accessible
try:
    from config import NO_CONTEXT_OPTION, APP_CONTEXT_FOLDER_PATH
except ImportError as e:
    st.error(f"Failed to import required config values. Ensure config.py exists: {e}")
    # Assign default values to allow app to potentially continue partially
    NO_CONTEXT_OPTION = "None"
    APP_CONTEXT_FOLDER_PATH = os.path.join(os.getcwd(), "app_context")


# --- Logging Utility ---
def log_message(message: str, level: str = "INFO"):
    """
    Appends a timestamped message to the session log list.

    Args:
        message: The message string to log.
        level: The log level (e.g., "INFO", "DEBUG", "ERROR", "WARNING").
               Currently just used for prefixing the message.
    """
    # Ensure the log list exists in session state
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    st.session_state.log_messages.append(log_entry)


# --- Dynamic Importing ---
def import_class(module_name: str, class_name: str):
    """
    Dynamically imports a class from a given module. Logs errors.

    Args:
        module_name: The name of the module (e.g., "langchain_openai").
        class_name: The name of the class (e.g., "ChatOpenAI").

    Returns:
        The imported class object or None if import fails.
    """
    log_message(f"Attempting to import {class_name} from {module_name}", "DEBUG")
    if not module_name or not class_name:
        log_message(f"Import failed: Module name ('{module_name}') or class name ('{class_name}') is empty.", "ERROR")
        st.error(f"Import failed: Module name ('{module_name}') or class name ('{class_name}') is empty.")
        return None
    try:
        module = importlib.import_module(module_name)
        imported_class = getattr(module, class_name)
        log_message(f"Successfully imported {class_name} from {module_name}", "DEBUG")
        return imported_class
    except ImportError:
        package_name = module_name.replace('_', '-')
        # Handle specific package names if needed
        if package_name == 'langchain-google-genai': package_name = 'langchain-google-genai'
        elif package_name == 'langchain-aws': package_name = 'langchain-aws boto3'
        elif package_name == 'langchain-anthropic': package_name = 'langchain-anthropic'
        elif package_name == 'langchain-groq': package_name = 'langchain-groq'
        error_msg = f"Failed to import `{class_name}` from `{module_name}`. Install required package: `pip install {package_name}`"
        log_message(error_msg, "ERROR")
        st.error(error_msg)
        return None
    except AttributeError:
        error_msg = f"Class `{class_name}` not found in module `{module_name}`."
        log_message(error_msg, "ERROR")
        st.error(error_msg)
        return None
    except Exception as e:
        error_msg = f"An unexpected error occurred during import of {class_name} from {module_name}: {e}"
        log_message(error_msg, "ERROR")
        st.error(error_msg)
        return None


# --- String/Filename Manipulation ---
def sanitize_filename(name: str, max_length: int = 200) -> str:
    """
    Removes invalid characters for filenames and optionally truncates.
    Also replaces spaces with underscores.

    Args:
        name: The input string.
        max_length: Maximum allowed length for the sanitized name.

    Returns:
        A sanitized string suitable for use in filenames or sheet names.
    """
    if not isinstance(name, str):
        name = str(name)
    name = name.replace(" ", "_")
    # Remove characters invalid for Windows/Mac filenames and Excel sheets
    name = re.sub(r'[<>:"/\\|?*\[\]\']', '', name)
    # Remove leading/trailing whitespace/underscores
    name = name.strip('._ ')
    # Truncate if necessary
    return name[:max_length]


# --- Context File Handling ---
def get_available_context_files() -> list[str]:
    """
    Scans the configured context folder for .yaml/.yml files. Logs warnings.

    Returns:
        A sorted list of available context file base names (without extension),
        including the "None" option.
    """
    context_files = [NO_CONTEXT_OPTION]
    folder = APP_CONTEXT_FOLDER_PATH
    log_message(f"Scanning for context files in: {folder}", "DEBUG")
    if os.path.exists(folder) and os.path.isdir(folder):
        try:
            for filename in os.listdir(folder):
                if filename.lower().endswith((".yaml", ".yml")):
                    base_name = os.path.splitext(filename)[0]
                    context_files.append(base_name)
                    log_message(f"Found context file: {filename} (base name: {base_name})", "DEBUG")
        except OSError as e:
            log_message(f"OS Error scanning context folder '{folder}': {e}", "WARNING")
            st.warning(f"Error scanning context folder '{folder}': {e}")
        except Exception as e:
            log_message(f"Unexpected error scanning context folder '{folder}': {e}", "WARNING")
            st.warning(f"Unexpected error scanning context folder '{folder}': {e}")
    else:
        log_message(f"Context folder not found or not a directory: {folder}", "INFO")
    # Return sorted unique list
    unique_sorted_files = sorted(list(set(context_files)))
    log_message(f"Available context files (incl. None): {unique_sorted_files}", "DEBUG")
    return unique_sorted_files

# --- JSON Parsing ---
def parse_json_output(llm_output: str, expected_type: type = list):
    """
    Attempts to parse JSON from LLM output, handling markdown code blocks. Logs details.

    Args:
        llm_output: The raw string output from the LLM.
        expected_type: The expected Python type (e.g., list, dict).

    Returns:
        The parsed JSON data (list or dict) or None if parsing fails
        or type mismatch.
    """
    log_message(f"Attempting to parse LLM output as {expected_type.__name__}", "DEBUG")
    if not llm_output:
        log_message("Parsing failed: LLM output is empty.", "WARNING")
        return None

    # Try finding JSON within markdown blocks first
    patterns = [
        r'```json\s*(.*)\s*```', # Standard markdown block
        r'```\s*(.*)\s*```'      # Generic code block (less specific)
    ]
    # Add raw patterns based on expected type AFTER code block patterns
    if expected_type == list:
        patterns.append(r'(\[.*\])') # Raw list, capture group 1
    elif expected_type == dict:
        patterns.append(r'(\{.*\})') # Raw dict, capture group 1

    json_str = None
    match_found = False
    for pattern in patterns:
        log_message(f"Trying regex pattern: {pattern}", "DEBUG")
        match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)
        if match:
            potential_json = match.group(1).strip()
            log_message(f"Regex match found, potential JSON: '{potential_json[:100]}...'", "DEBUG")
            # Basic check if it looks like the expected type
            looks_like_list = expected_type == list and potential_json.startswith('[') and potential_json.endswith(']')
            looks_like_dict = expected_type == dict and potential_json.startswith('{') and potential_json.endswith('}')

            if looks_like_list or looks_like_dict:
                 json_str = potential_json
                 match_found = True
                 log_message("Potential JSON structure matches expected type.", "DEBUG")
                 break # Use the first valid-looking match from patterns
            else:
                 log_message("Potential JSON structure does NOT match expected type, continuing search.", "DEBUG")

    if not match_found:
        # If no markdown or raw match, try parsing the whole string if it looks right
        log_message("No regex match found, trying full string parsing.", "DEBUG")
        trimmed_output = llm_output.strip()
        if expected_type == list and trimmed_output.startswith('[') and trimmed_output.endswith(']'):
            json_str = trimmed_output
            log_message("Using full trimmed output as potential list.", "DEBUG")
        elif expected_type == dict and trimmed_output.startswith('{') and trimmed_output.endswith('}'):
            json_str = trimmed_output
            log_message("Using full trimmed output as potential dict.", "DEBUG")

    if not json_str:
         log_message(f"Could not extract valid {expected_type.__name__} structure from LLM output.", "WARNING")
         st.warning(f"Could not find valid {expected_type.__name__} structure in LLM output.")
         # Optional: Log more of the raw output for debugging
         # log_message(f"RAW LLM OUTPUT for failed parse:\n{llm_output}", "DEBUG")
         return None

    try:
        log_message(f"Attempting json.loads on extracted string: '{json_str[:100]}...'", "DEBUG")
        parsed_data = json.loads(json_str)
        log_message("json.loads successful.", "DEBUG")
        if isinstance(parsed_data, expected_type):
            log_message(f"Parsed data type matches expected type ({expected_type.__name__}).", "INFO")
            return parsed_data
        else:
            log_message(f"Parsed data type is {type(parsed_data).__name__}, expected {expected_type.__name__}.", "WARNING")
            st.warning(f"LLM output parsed, but type is {type(parsed_data).__name__}, expected {expected_type.__name__}.")
            return None
    except json.JSONDecodeError as e:
        log_message(f"JSON Parsing Error: {e}. Invalid JSON string: '{json_str[:200]}...'", "ERROR")
        st.error(f"Failed to parse JSON from LLM output: {e}")
        # Consider showing only part of the invalid string in UI
        st.text_area("Invalid JSON String (for debugging):", json_str[:500]+"...", height=100)
        return None
    except Exception as e:
        log_message(f"An unexpected error occurred during JSON parsing: {e}", "ERROR")
        st.error(f"An unexpected error occurred during JSON parsing: {e}")
        return None
