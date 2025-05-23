# llm_integration_core.py
# Main module for LangChain setup, LLM interactions (identification, RAG, refactoring).
# Adapted for a backend environment.

import yaml
import os
import json
import re
import ast
from typing import Dict, Any, Tuple, List, Optional

# Langchain Core Imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# Import config and utilities
try:
    import config
    from config import (
        LLM_PROVIDER_CONFIG, FALLBACK_EMBEDDING_PROVIDERS,
        CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_SEARCH_K,
        EXCEL_EXPECTED_COLUMNS
    )
    from helper.utils import parse_json_output, log_message
    log_message("Successfully imported config and helper.utils in llm_integration_core.", "INFO")
except ImportError as e:
    print(f"CRITICAL: Failed to import required modules (config, helper.utils) in llm_integration_core: {e}")
    if 'log_message' not in globals():
        def log_message(msg, level, **kwargs): print(f"LLM_CORE_FALLBACK_LOGGER [{level}] {msg}")
    log_message(f"CRITICAL: Failed to import required modules (config, helper.utils) in llm_integration_core: {e}", "ERROR")
    raise

# Provider-Specific Initialization Imports
try:
    from llm_providers.llm_openai import _initialize_openai
    from llm_providers.llm_gemini import _initialize_gemini
    from llm_providers.llm_openrouter import _initialize_openrouter
    from llm_providers.llm_vertexai import _initialize_vertexai
    from llm_providers.llm_claude import _initialize_claude
    from llm_providers.llm_bedrock import _initialize_bedrock
    from llm_providers.llm_groq import _initialize_groq
    from llm_providers.llm_ollama import _initialize_ollama
    from llm_providers.llm_embeddings_utils import _get_fallback_embeddings
    log_message("Successfully imported provider initialization functions from 'llm_providers'.", "DEBUG")
except ImportError as e:
    log_message(f"CRITICAL: Failed to import provider logic from 'llm_providers': {e}", "ERROR", exc_info=True)
    raise

def get_llm_and_embeddings(provider: str, model_name: str, credentials: Dict, fallback_openai_key: str) -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    log_message(f"Getting LLM and Embeddings for provider: {provider}, model: {model_name}", "INFO")
    llm: Optional[BaseChatModel] = None
    embeddings: Optional[Embeddings] = None
    config_dict = LLM_PROVIDER_CONFIG.get(provider)
    if not config_dict:
        log_message(f"Invalid provider selected: {provider}. No configuration found.", "ERROR")
        return None, None
    init_functions = {
        "OpenAI": _initialize_openai, 
        "Gemini": _initialize_gemini, 
        "Claude": _initialize_claude,
        "AWS Bedrock": _initialize_bedrock, 
        "Groq": _initialize_groq, 
        "Ollama": _initialize_ollama,
        "OpenRouter": _initialize_openrouter,
        "Vertex AI": _initialize_vertexai
    }
    init_func = init_functions.get(provider)
    if not init_func:
        log_message(f"Initialization function not found for provider: {provider}", "ERROR")
        return None, None
    try:
        llm, embeddings = init_func(config_dict, credentials, model_name)
    except Exception as e:
        log_message(f"Unexpected error calling initialization function for {provider}: {e}", "ERROR", exc_info=True)
        return None, None
    is_fallback_provider = provider in FALLBACK_EMBEDDING_PROVIDERS
    if is_fallback_provider and llm and not embeddings:
        log_message(f"Provider {provider} requires fallback embeddings. Attempting initialization.", "INFO")
        embeddings = _get_fallback_embeddings(fallback_openai_key)
        if not embeddings:
            log_message(f"Failed to initialize fallback OpenAI embeddings for {provider}.", "WARNING")
            return llm, None
        else:
             log_message(f"Successfully initialized fallback OpenAI embeddings for {provider}.", "INFO")
    if not llm:
        log_message(f"LLM initialization final check failed for {provider}.", "ERROR")
        return None, None
    if not embeddings and not is_fallback_provider:
        log_message(f"Embeddings init failed for non-fallback provider {provider}. RAG may not work.", "WARNING")
        return llm, None
    log_message(f"LLM and Embeddings initialized successfully for {provider}.", "INFO")
    return llm, embeddings

def check_credentials(provider: str, credentials: Dict, fallback_key: str, require_fallback_for_rag: bool) -> Tuple[bool, str]:
    log_message(f"Checking credentials for provider: {provider}", "DEBUG")
    config_dict = LLM_PROVIDER_CONFIG.get(provider)
    if not config_dict:
        return False, f"Invalid provider: '{provider}'. No config found."
    missing = []
    required_creds = config_dict.get("credentials", [])
    for key in required_creds:
        if provider == "Ollama" and key in ["base_url", "model"]: continue
        if provider == "AWS Bedrock" and key == "aws_session_token":
            key_id = credentials.get("aws_access_key_id", "")
            token = credentials.get("aws_session_token", "")
            if key_id.startswith("ASIA") and not token:
                missing.append("AWS Session Token (for temporary credentials)")
            continue
        if not credentials.get(key, "").strip():
            missing.append(key.replace("_", " ").title())
    if require_fallback_for_rag and provider in FALLBACK_EMBEDDING_PROVIDERS and not fallback_key.strip():
        missing.append("OpenAI API Key (for RAG Fallback Embeddings)")
    if missing:
        error_msg = f"Missing credentials/settings for {provider}: {', '.join(missing)}."
        log_message(error_msg, "WARNING")
        return False, error_msg
    return True, ""

def get_prompt_template(provider_name: str, template_key: str) -> str:
    if 'config' not in globals():
        log_message("Config module not loaded in get_prompt_template.", "ERROR")
        raise ImportError("Configuration module 'config' not loaded.")
    provider_cfg = config.LLM_PROVIDER_CONFIG.get(provider_name, {})
    provider_prompts = provider_cfg.get("prompt_templates", {})
    if template_key in provider_prompts:
        return provider_prompts[template_key]
    default_template_name = f"{template_key}_PROMPT_TEMPLATE"
    if hasattr(config, default_template_name):
        return getattr(config, default_template_name)
    error_msg = f"Prompt template '{template_key}' not found for '{provider_name}' or in defaults."
    log_message(error_msg, "ERROR")
    return f"ERROR: Prompt template '{template_key}' is missing."

def _extract_list_from_llm_output(raw_output: str) -> Optional[str]:
    log_message("Attempting to extract list structure from raw LLM output...", "DEBUG")
    if not raw_output or not raw_output.strip():
        return None
    code_block_patterns = [
        r"```(?:python|json)\s*(\[.*?\])\s*```",
        r"```\s*(\[.*?\])\s*```"
    ]
    for pattern in code_block_patterns:
        match = re.search(pattern, raw_output, re.DOTALL | re.IGNORECASE)
        if match:
            extracted_from_block = match.group(1).strip()
            try: json.loads(extracted_from_block); return extracted_from_block
            except json.JSONDecodeError:
                try: ast.literal_eval(extracted_from_block); return extracted_from_block
                except (ValueError, SyntaxError): continue
    direct_match = re.search(r'(\[.*?\])', raw_output, re.DOTALL)
    if direct_match:
        extracted_direct = direct_match.group(1).strip()
        try: json.loads(extracted_direct); return extracted_direct
        except json.JSONDecodeError:
            try: ast.literal_eval(extracted_direct); return extracted_direct
            except (ValueError, SyntaxError): pass
    bullet_items = []
    bullet_pattern = re.compile(r'^\s*[\*\-]\s+(.*)', re.MULTILINE)
    matches = bullet_pattern.findall(raw_output)
    if matches:
        for item in matches:
            cleaned_item = item.strip().strip("'\"`")
            if cleaned_item: bullet_items.append(json.dumps(cleaned_item))
        if bullet_items:
            list_string = f"[{', '.join(bullet_items)}]"
            try: json.loads(list_string); return list_string
            except json.JSONDecodeError:
                 log_message(f"Constructed list from bullets failed JSON validation: {list_string}", "ERROR")
    log_message("Could not extract a valid list structure from LLM output.", "WARNING")
    return None

def identify_applications(text: str, llm: BaseChatModel, provider_name: str) -> List[str]:
    log_message(f"Starting app identification (provider: {provider_name})...", "INFO")
    if not text or not text.strip(): log_message("Identify failed: Input text empty.", "ERROR"); return []
    if not llm: log_message("Identify failed: LLM not initialized.", "ERROR"); return []
    try:
        template_key = "IDENTIFY_APP"
        prompt_str = get_prompt_template(provider_name, template_key)
        if prompt_str.startswith("Error:"): log_message(f"Identify failed: {prompt_str}", "ERROR"); return []
        app_prompt = ChatPromptTemplate.from_template(prompt_str)
        app_chain = app_prompt | llm | StrOutputParser()
        result_str = app_chain.invoke({"text": text})
        log_message(f"Raw LLM output for identification: {result_str[:200]}...", "DEBUG")
        if not result_str or not result_str.strip(): log_message("LLM returned empty for identification.", "WARNING"); return []
        extracted_list_str = _extract_list_from_llm_output(result_str)
        parsed_apps = parse_json_output(extracted_list_str, expected_type=list) if extracted_list_str else parse_json_output(result_str, expected_type=list)
        if parsed_apps is None:
            log_message("Failed to parse app list from LLM.", "WARNING")
            if extracted_list_str is None and not result_str.strip().startswith(("[", "{")) and ',' in result_str:
                possible_apps = [app.strip().strip("'\"`") for app in result_str.split(',') if app.strip()]
                if possible_apps:
                    log_message(f"Using comma parsing fallback for identification. Found: {possible_apps}", "INFO")
                    return sorted(list(set(app for app in possible_apps if app)))
            return []
        cleaned_apps = [str(app).strip() for app in parsed_apps if isinstance(app, (str, int, float)) and str(app).strip()]
        if not cleaned_apps: log_message("Parsed app list empty or non-string items.", "WARNING"); return []
        final_apps = sorted(list(set(cleaned_apps)))
        log_message(f"Identification successful. Apps: {final_apps}", "INFO")
        return final_apps
    except Exception as e:
        log_message(f"Exception during app identification: {e}", "ERROR", exc_info=True)
        return []

def generate_test_cases(
    text: str, selected_apps: List[str],
    uploaded_context_files_content: Dict[str, List[str]],
    llm: BaseChatModel, embeddings: Embeddings, provider_name: str
) -> Dict[str, Any]:
    log_message(f"Generate TCs (provider: {provider_name})...", "DEBUG")
    results = {}
    if not selected_apps: return results
    if not text or not text.strip(): return {app: "Error: Source text empty." for app in selected_apps}
    if not llm: return {app: "Error: LLM not initialized." for app in selected_apps}
    if not embeddings: return {app: "Error: Embeddings not initialized (RAG)." for app in selected_apps}
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = text_splitter.split_text(text)
        if not splits: raise ValueError("Main text splitting resulted in zero chunks.")
        vectorstore = FAISS.from_texts(texts=splits, embedding=embeddings)
        if not vectorstore: raise ValueError("Vector store creation failed.")
    except Exception as e:
        log_message(f"Vector store creation error: {e}", "ERROR", exc_info=True)
        return {app: f"Error creating embeddings/vector store: {e}" for app in selected_apps}
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_SEARCH_K})
    template_key = "GENERATE_TC"
    prompt_str = get_prompt_template(provider_name, template_key)
    if prompt_str.startswith("Error:"): return {app: prompt_str for app in selected_apps}
    if not isinstance(EXCEL_EXPECTED_COLUMNS, list) or not EXCEL_EXPECTED_COLUMNS:
        return {app: "Config error: EXCEL_EXPECTED_COLUMNS missing." for app in selected_apps}
    tc_fields = ", ".join([f"`{col}`" for col in EXCEL_EXPECTED_COLUMNS])
    try:
        formatted_prompt_str = prompt_str.format(field_names=tc_fields, context="{context}", input="{input}")
    except KeyError as e:
        return {app: f"Prompt template '{template_key}' missing placeholders: {e}" for app in selected_apps}
    test_case_prompt = ChatPromptTemplate.from_template(formatted_prompt_str)
    document_chain = create_stuff_documents_chain(llm, test_case_prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    for app_name in selected_apps:
        try:
            additional_context_str = ""
            app_specific_context_list = uploaded_context_files_content.get(app_name, [])
            if app_specific_context_list:
                for idx, content_str in enumerate(app_specific_context_list):
                    additional_context_str += f"\n\n--- Additional Context {idx+1} ---\n{content_str}\n--- End Context ---\n"
            input_query = (f"Generate detailed test cases for application: '{app_name}'. "
                           f"Use retrieved requirements and this additional context: {additional_context_str}")
            response = retrieval_chain.invoke({"input": input_query})
            if isinstance(response, dict) and "answer" in response and response["answer"]:
                answer_str = response["answer"].strip()
                parsed_cases = parse_json_output(answer_str, expected_type=list)
                if parsed_cases is not None and isinstance(parsed_cases, list):
                    results[app_name] = [item for item in parsed_cases if isinstance(item, dict)]
                else: results[app_name] = "Error: Failed to parse JSON list of TCs from LLM."
            else: results[app_name] = "Error: LLM gave no answer or unexpected response format."
        except Exception as e:
            log_message(f"Exception for app '{app_name}' in TC generation: {e}", "ERROR", exc_info=True)
            results[app_name] = f"Error: Generation failed - {e}"
    return results

def refactor_single_test_case(
    app_name: str, tc_id: str, instructions: str,
    original_tc_data: Dict, llm: BaseChatModel, provider_name: str
) -> Optional[Dict]:
    log_message(f"Refactor TC '{tc_id}' for '{app_name}'...", "INFO")
    if not llm: return None
    if not original_tc_data or not isinstance(original_tc_data, dict): return None
    if not instructions or not instructions.strip(): return None
    try:
        original_json_str = json.dumps(original_tc_data, separators=(',', ':'))
        template_key = "REFACTOR_TC"
        prompt_str = get_prompt_template(provider_name, template_key)
        if prompt_str.startswith("Error:"): return None
        prompt = ChatPromptTemplate.from_template(prompt_str)
        chain = prompt | llm | StrOutputParser()
        response_str = chain.invoke({
            "tc_id": tc_id, "original_tc_json": original_json_str, "user_instructions": instructions
        })
        if not response_str or not response_str.strip(): return None
        updated_tc_data = parse_json_output(response_str, expected_type=dict)
        if updated_tc_data is None or not isinstance(updated_tc_data, dict): return None
        original_id_val = original_tc_data.get("Test Case ID", tc_id)
        if updated_tc_data.get("Test Case ID") != original_id_val:
            updated_tc_data["Test Case ID"] = original_id_val
        return updated_tc_data
    except Exception as e:
        log_message(f"Exception during refactor TC '{tc_id}': {e}", "ERROR", exc_info=True)
        return None

def refactor_all_test_cases(
    app_name: str, instructions: str, original_tc_list: List[Dict],
    llm: BaseChatModel, provider_name: str
) -> Optional[List[Dict]]:
    log_message(f"Bulk refactor for '{app_name}'...", "INFO")
    if not llm: return None
    if not original_tc_list or not isinstance(original_tc_list, list) or not all(isinstance(tc, dict) for tc in original_tc_list): return None
    if not instructions or not instructions.strip(): return None
    try:
        original_list_json_str = json.dumps(original_tc_list, separators=(',', ':'))
        template_key = "REFACTOR_ALL_TC"
        prompt_str = get_prompt_template(provider_name, template_key)
        if prompt_str.startswith("Error:"): return None
        prompt = ChatPromptTemplate.from_template(prompt_str)
        chain = prompt | llm | StrOutputParser()
        response_str = chain.invoke({
            "app_name": app_name, "original_tc_list_json": original_list_json_str, "user_instructions": instructions
        })
        extracted_list_str = _extract_list_from_llm_output(response_str)
        updated_tc_list = parse_json_output(extracted_list_str, expected_type=list) if extracted_list_str else parse_json_output(response_str, expected_type=list)
        if updated_tc_list is None or not isinstance(updated_tc_list, list): return None
        return [tc for tc in updated_tc_list if isinstance(tc, dict)]
    except Exception as e:
        log_message(f"Exception during bulk refactor for '{app_name}': {e}", "ERROR", exc_info=True)
        return None

def perform_ai_test_case_review(
    main_requirements_text: str, additional_context_str: str, existing_test_cases: List[Dict],
    llm: BaseChatModel, provider_name: str
) -> Optional[Dict]:
    log_message(f"AI TC review (provider: {provider_name})...", "INFO")
    if not llm: return None
    if not main_requirements_text: return None
    if not isinstance(existing_test_cases, list): return None
    try:
        existing_test_cases_json_str = json.dumps(existing_test_cases, separators=(',', ':'))
        if not isinstance(EXCEL_EXPECTED_COLUMNS, list) or not EXCEL_EXPECTED_COLUMNS: return None
        field_names_str = ", ".join([f"`{col}`" for col in EXCEL_EXPECTED_COLUMNS])
        template_key = "AI_REVIEW_TC"
        prompt_str = get_prompt_template(provider_name, template_key)
        if prompt_str.startswith("Error:"): return None
        formatted_prompt_str = prompt_str.format(field_names=field_names_str)
        prompt = ChatPromptTemplate.from_template(formatted_prompt_str)
        chain = prompt | llm | StrOutputParser()
        response_str = chain.invoke({
            "main_requirements": main_requirements_text,
            "additional_context": additional_context_str,
            "existing_test_cases_json": existing_test_cases_json_str
        })
        review_results_dict = parse_json_output(response_str, expected_type=dict)
        if review_results_dict is None or not isinstance(review_results_dict, dict): return None
        expected_top_keys = ["coverage_summary", "newly_suggested_test_cases", "modified_test_cases_suggestions", "identified_duplicates"]
        for key in expected_top_keys:
            if key not in review_results_dict:
                if key.endswith("_summary"): review_results_dict[key] = "Summary not provided."
                else: review_results_dict[key] = []
        return review_results_dict
    except Exception as e:
        log_message(f"Exception during AI TC review: {e}", "ERROR", exc_info=True)
        return None

def apply_ai_review_changes_logic(
    app_name: str,
    existing_test_cases: List[Dict[str, Any]],
    ai_review_suggestions_processed: Dict[str, Any],
    user_decisions: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Applies user-accepted AI review suggestions to a list of test cases.
    This logic is adapted from the original Streamlit application.
    """
    log_message(f"Applying AI review changes for app: {app_name}", "INFO")
    
    current_tcs_for_app = [dict(tc) for tc in existing_test_cases] # Work on a mutable copy

    new_tcs_applied_count = 0
    modifications_applied_count = 0
    duplicates_removed_count = 0

    # Determine the next available Test Case ID suffix
    max_id_num = 0
    for tc in current_tcs_for_app:
        if isinstance(tc, dict) and "Test Case ID" in tc:
            match = re.search(r'_(\d+)$', tc["Test Case ID"])
            if match:
                max_id_num = max(max_id_num, int(match.group(1)))
    next_id_counter = max_id_num + 1

    # Process accepted new test cases
    new_suggestions = ai_review_suggestions_processed.get("newly_suggested_test_cases", [])
    for i, suggested_tc_data in enumerate(new_suggestions):
        suggestion_id = f"new_{app_name}_{i}" # This key format must match how frontend generates it for user_decisions
        if user_decisions.get(suggestion_id) == "accept":
            new_tc_to_add = dict(suggested_tc_data)
            new_tc_to_add["Test Case ID"] = f"{app_name}_TC_{next_id_counter}"
            next_id_counter += 1
            current_tcs_for_app.append(new_tc_to_add)
            new_tcs_applied_count += 1
            log_message(f"Applied new TC: {new_tc_to_add['Test Case ID']}", "INFO")

    # Process accepted modified test cases
    modified_suggestions = ai_review_suggestions_processed.get("modified_test_cases_suggestions", [])
    if modified_suggestions:
        for mod_suggestion_details in modified_suggestions:
            original_tc_id_to_modify = mod_suggestion_details.get("original_test_case_id")
            # Key format for user_decisions must match frontend
            suggestion_key = f"mod_{app_name}_{original_tc_id_to_modify}" 

            if user_decisions.get(suggestion_key) == "accept":
                suggested_data = mod_suggestion_details.get("suggested_test_case_data")
                if not suggested_data or not isinstance(suggested_data, dict):
                    log_message(f"Skipping mod for TC ID '{original_tc_id_to_modify}': data invalid.", "WARNING")
                    continue

                found_and_modified = False
                for i, existing_tc in enumerate(current_tcs_for_app):
                    if isinstance(existing_tc, dict) and existing_tc.get("Test Case ID") == original_tc_id_to_modify:
                        final_modified_data = dict(suggested_data)
                        final_modified_data["Test Case ID"] = original_tc_id_to_modify # Preserve original ID
                        current_tcs_for_app[i] = final_modified_data
                        modifications_applied_count += 1
                        found_and_modified = True
                        log_message(f"Applied modification for TC ID: {original_tc_id_to_modify}", "INFO")
                        break
                if not found_and_modified:
                    log_message(f"Could not find original TC ID '{original_tc_id_to_modify}' for modification.", "WARNING")
    
    # Process accepted duplicate resolutions
    duplicate_suggestions_groups = ai_review_suggestions_processed.get("identified_duplicates", [])
    if duplicate_suggestions_groups:
        tc_ids_to_remove_overall = set()
        for dup_group_details in duplicate_suggestions_groups:
            group_id = dup_group_details.get("duplicate_group_id")
            tc_ids_in_group = dup_group_details.get("test_case_ids", [])
            
            if not group_id or not tc_ids_in_group:
                continue

            # Key format for user_decisions must match frontend
            decision_key = f"dup_{app_name}_{group_id}"
            tc_id_to_keep = user_decisions.get(decision_key)

            if tc_id_to_keep and tc_id_to_keep != "Resolve Later / No Action": # Assuming "Resolve Later / No Action" is a possible value from frontend
                for tc_id_in_group_member in tc_ids_in_group:
                    if tc_id_in_group_member != tc_id_to_keep:
                        tc_ids_to_remove_overall.add(tc_id_in_group_member)
                log_message(f"Dup group '{group_id}': Keeping '{tc_id_to_keep}', removing others.", "INFO")
        
        if tc_ids_to_remove_overall:
            original_len = len(current_tcs_for_app)
            current_tcs_for_app[:] = [tc for tc in current_tcs_for_app if tc.get("Test Case ID") not in tc_ids_to_remove_overall]
            duplicates_removed_count = original_len - len(current_tcs_for_app)
            if duplicates_removed_count > 0:
                 log_message(f"Removed {duplicates_removed_count} duplicate TCs for app '{app_name}'.", "INFO")

    summary_messages = []
    if new_tcs_applied_count > 0: summary_messages.append(f"{new_tcs_applied_count} new test case(s) added")
    if modifications_applied_count > 0: summary_messages.append(f"{modifications_applied_count} existing test case(s) modified")
    if duplicates_removed_count > 0: summary_messages.append(f"{duplicates_removed_count} duplicate test case(s) removed")
    
    final_summary = ", ".join(summary_messages) if summary_messages else "No changes were applied based on user decisions."
    log_message(f"AI Review Apply Summary for '{app_name}': {final_summary}", "INFO")
    
    return current_tcs_for_app, final_summary
