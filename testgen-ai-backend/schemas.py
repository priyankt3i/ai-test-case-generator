from pydantic import BaseModel
from typing import List, Optional, Dict, Any # Added Any

class LLMProviderDetail(BaseModel):
    models: List[str]
    credentials: List[str]
    llm_module: str
    llm_class: str
    embeddings_module: Optional[str] = None
    embeddings_class: Optional[str] = None
    notes: Optional[str] = None
    prompt_templates: Optional[Dict[str, str]] = None
    embedding_model_ids: Optional[List[str]] = None # For Bedrock
    embeddings_model_id: Optional[str] = None # For Gemini

class LLMProviderConfigResponse(BaseModel):
    providers: Dict[str, LLMProviderDetail]

class RequirementsUploadResponse(BaseModel):
    filename: str
    content_type: Optional[str] = None
    extracted_text: Optional[str] = None
    error: Optional[str] = None

class ExportRequest(BaseModel):
    # Data structure: { "AppName1": [tc1_dict, tc2_dict], "AppName2": [...] }
    test_cases_data: Dict[str, List[Dict[str, Any]]] 
    filename: str = "test_cases_export.xlsx"

class GenerateTestCasesRequest(BaseModel):
    main_requirements_text: str
    selected_applications: List[str]
    # For each app, a list of strings, where each string is the content of a context file
    application_contexts: Dict[str, List[str]] = {}
    llm_provider: str
    model_name: str
    api_credentials: Dict[str, str]
    openai_fallback_api_key: Optional[str] = None

class GenerateTestCasesResponse(BaseModel):
    # The result from llm_integration_core.generate_test_cases is Dict[str, Any]
    # where Any can be List[Dict[str, str]] (test cases) or str (error message for an app)
    results: Dict[str, Any] = {}
    error: Optional[str] = None # For overall errors, not per-app errors

class RefactorSingleTestCaseRequest(BaseModel):
    app_name: str
    tc_id: str
    instructions: str
    original_tc_data: Dict[str, Any]
    llm_provider: str
    model_name: str
    api_credentials: Dict[str, str]
    openai_fallback_api_key: Optional[str] = None

class RefactorSingleTestCaseResponse(BaseModel):
    refactored_tc_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class RefactorAllTestCasesRequest(BaseModel):
    instructions: str
    original_tc_list: List[Dict[str, Any]] # Client sends the list of TCs for the app
    llm_provider: str
    model_name: str
    api_credentials: Dict[str, str]
    openai_fallback_api_key: Optional[str] = None

class RefactorAllTestCasesResponse(BaseModel):
    app_name: str
    refactored_test_cases: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

class AIReviewRequest(BaseModel):
    main_requirements_text: str
    additional_context_str: str # Combined text from all context docs for the app
    existing_test_cases: List[Dict[str, Any]]
    llm_provider: str
    model_name: str
    api_credentials: Dict[str, str]
    openai_fallback_api_key: Optional[str] = None

class AIReviewResponse(BaseModel):
    app_name: str
    # Structure from llm_core.perform_ai_test_case_review:
    # Dict with keys like "coverage_summary", "newly_suggested_test_cases", etc.
    review_results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ApplyAIReviewRequest(BaseModel):
    existing_test_cases: List[Dict[str, Any]]
    # This is the structured output from the perform_ai_test_case_review function
    ai_review_suggestions_processed: Dict[str, Any] 
    # Maps suggestion_id (e.g., "new_AppName_0", "mod_AppName_TC1", "dup_AppName_group1") 
    # to user's choice (e.g., "accept", "reject", or the ID of the test case to keep for duplicates)
    user_decisions: Dict[str, str] 

class ApplyAIReviewResponse(BaseModel):
    app_name: str
    updated_test_cases: List[Dict[str, Any]]
    summary_message: str # e.g., "X new TCs added, Y modified, Z duplicates removed."
    error: Optional[str] = None

class IdentifyAppRequest(BaseModel):
    extracted_text: str
    llm_provider: str
    model_name: str
    api_credentials: Dict[str, str]
    openai_fallback_api_key: Optional[str] = None

class IdentifyAppResponse(BaseModel):
    identified_applications: List[str] = []
    error: Optional[str] = None

class ContextUploadResponse(BaseModel):
    app_name: str
    filename: str
    content_type: Optional[str] = None
    extracted_text: Optional[str] = None
    error: Optional[str] = None
