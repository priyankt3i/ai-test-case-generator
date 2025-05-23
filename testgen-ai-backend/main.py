from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict
import io

# Assuming config.py is in the same directory or Python path
import config as app_config
from schemas import (
    LLMProviderDetail, LLMProviderConfigResponse,
    RequirementsUploadResponse,
    IdentifyAppRequest, IdentifyAppResponse,
    ContextUploadResponse,
    GenerateTestCasesRequest, GenerateTestCasesResponse,
    RefactorSingleTestCaseRequest, RefactorSingleTestCaseResponse,
    RefactorAllTestCasesRequest, RefactorAllTestCasesResponse,
    AIReviewRequest, AIReviewResponse,
    ApplyAIReviewRequest, ApplyAIReviewResponse,
    ExportRequest # Added ExportRequest
)
# Assuming helper.file_processing.py is in the helper directory
from helper.file_processing import extract_text_from_file
from helper.excel_export import export_test_cases_to_excel_bytes # Added excel export
# Assuming llm_integration_core.py is in the same directory or Python path
import llm_integration_core as llm_core
from helper.utils import log_message # For logging within endpoints

app = FastAPI(title="TestGen AI Backend", version="0.1.0")

# CORS Configuration
origins = [
    "http://localhost",         # Common base for local dev
    "http://localhost:3000",    # Common Create React App port
    "http://localhost:5173",    # Common Vite default port
    # Add your deployed frontend URL here later, e.g., "https://your-frontend.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

@app.get("/")
async def root():
    return {"message": "TestGen AI Backend is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/config/llm-providers", response_model=LLMProviderConfigResponse)
async def get_llm_providers_config():
    """
    Returns the configuration for all available LLM providers.
    """
    # Directly use the structure from config.LLM_PROVIDER_CONFIG
    # FastAPI will validate it against LLMProviderConfigResponse
    # and its nested LLMProviderDetail models.
    # We need to ensure the keys in app_config.LLM_PROVIDER_CONFIG
    # match the fields in LLMProviderDetail.
    # If there are extra keys in app_config.LLM_PROVIDER_CONFIG that are not in LLMProviderDetail,
    # Pydantic will ignore them by default if not part of the model.
    # If keys are missing and not Optional in Pydantic model, it will error.
    
    # Constructing the response by explicitly mapping to ensure compatibility
    # and handle any potential discrepancies or transformations if needed.
    # For now, a direct pass-through is attempted, relying on Pydantic validation.
    
    # A more robust way is to iterate and construct:
    providers_data: Dict[str, LLMProviderDetail] = {}
    for provider_name, details_dict in app_config.LLM_PROVIDER_CONFIG.items():
        # Pydantic will raise a validation error if details_dict doesn't match LLMProviderDetail
        try:
            providers_data[provider_name] = LLMProviderDetail(**details_dict)
        except Exception as e:
            # Log this error in a real app
            print(f"Error processing provider {provider_name}: {e}")
    # Optionally skip this provider or return an error
            continue

    return LLMProviderConfigResponse(providers=providers_data)

@app.post("/api/upload/requirements", response_model=RequirementsUploadResponse)
async def upload_requirements_document(file: UploadFile = File(...)):
    """
    Uploads a requirements document (.docx) and extracts its text.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .docx files are accepted.")

    try:
        # The extract_text_from_file function from helper.file_processing
        # expects a Streamlit UploadedFile-like object.
        # FastAPI's UploadFile is similar enough for getvalue() and name.
        # If extract_text_from_file relies on other Streamlit-specific attributes,
        # it might need adjustment or a wrapper. For now, assume it works.
        extracted_text = extract_text_from_file(file)

        if extracted_text is None:
            # This could happen if mammoth is not available or extraction fails internally
            return RequirementsUploadResponse(
                filename=file.filename,
                content_type=file.content_type,
                error="Failed to extract text from the document. The file might be corrupted or an internal error occurred."
            )

        return RequirementsUploadResponse(
            filename=file.filename,
            content_type=file.content_type,
            extracted_text=extracted_text
        )
    except HTTPException:
        raise # Re-raise HTTPException
    except Exception as e:
        # Log the exception e
        log_message(f"Error during file upload processing: {e}", "ERROR", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred processing the file: {str(e)}")

@app.post("/api/actions/identify-applications", response_model=IdentifyAppResponse)
async def identify_applications_action(request: IdentifyAppRequest):
    """
    Identifies applications from the provided text using the specified LLM.
    """
    log_message(f"Identify applications request received for provider: {request.llm_provider}", "INFO")

    # 1. Check credentials
    creds_ok, creds_msg = llm_core.check_credentials(
        provider=request.llm_provider,
        credentials=request.api_credentials,
        fallback_key=request.openai_fallback_api_key or "",
        require_fallback_for_rag=False # Not strictly RAG for identification
    )
    if not creds_ok:
        log_message(f"Credential check failed for identify_applications: {creds_msg}", "WARNING")
        return IdentifyAppResponse(error=f"Credential check failed: {creds_msg}")

    # 2. Initialize LLM
    # Note: identify_applications in llm_integration_core expects the LLM object directly.
    # We need to get the LLM instance here.
    llm, _ = llm_core.get_llm_and_embeddings(
        provider=request.llm_provider,
        model_name=request.model_name,
        credentials=request.api_credentials,
        fallback_openai_key=request.openai_fallback_api_key or ""
    )

    if not llm:
        error_msg = f"Failed to initialize LLM {request.model_name} for provider {request.llm_provider}."
        log_message(error_msg, "ERROR")
        return IdentifyAppResponse(error=error_msg)

    # 3. Call the core identification logic
    try:
        identified_apps = llm_core.identify_applications(
            text=request.extracted_text,
            llm=llm,
            provider_name=request.llm_provider
        )
        log_message(f"Identified applications: {identified_apps}", "INFO")
        return IdentifyAppResponse(identified_applications=identified_apps)
    except Exception as e:
        error_msg = f"An unexpected error occurred during application identification: {str(e)}"
        log_message(error_msg, "ERROR", exc_info=True)
        return IdentifyAppResponse(error=error_msg)

@app.post("/api/upload/context/{app_name}", response_model=ContextUploadResponse)
async def upload_context_document(app_name: str, file: UploadFile = File(...)):
    """
    Uploads a context document for a specific application and extracts its text.
    Accepts .txt, .md, .docx, .xlsx, .json, .yaml files.
    """
    log_message(f"Context file upload request for app: {app_name}, file: {file.filename}", "INFO")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    allowed_extensions = (".txt", ".md", ".docx", ".xlsx", ".json", ".yaml", ".yml")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}")

    try:
        extracted_text = extract_text_from_file(file)

        if extracted_text is None:
            return ContextUploadResponse(
                app_name=app_name,
                filename=file.filename,
                content_type=file.content_type,
                error="Failed to extract text from the context document. File might be corrupted or type unsupported by extraction."
            )

        return ContextUploadResponse(
            app_name=app_name,
            filename=file.filename,
            content_type=file.content_type,
            extracted_text=extracted_text
        )
    except HTTPException:
        raise # Re-raise HTTPException
    except Exception as e:
        log_message(f"Error during context file upload processing for {app_name}: {e}", "ERROR", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred processing the context file: {str(e)}")

@app.post("/api/actions/generate-test-cases", response_model=GenerateTestCasesResponse)
async def generate_test_cases_action(request: GenerateTestCasesRequest):
    """
    Generates test cases based on main requirements, selected applications, and their context data.
    """
    log_message(f"Generate test cases request for apps: {request.selected_applications}, provider: {request.llm_provider}", "INFO")

    # 1. Check credentials (require_fallback_for_rag is True for generation)
    creds_ok, creds_msg = llm_core.check_credentials(
        provider=request.llm_provider,
        credentials=request.api_credentials,
        fallback_key=request.openai_fallback_api_key or "",
        require_fallback_for_rag=True
    )
    if not creds_ok:
        log_message(f"Credential check failed for generate_test_cases: {creds_msg}", "WARNING")
        return GenerateTestCasesResponse(error=f"Credential check failed: {creds_msg}")

    # 2. Initialize LLM and Embeddings
    llm, embeddings = llm_core.get_llm_and_embeddings(
        provider=request.llm_provider,
        model_name=request.model_name,
        credentials=request.api_credentials,
        fallback_openai_key=request.openai_fallback_api_key or ""
    )

    if not llm:
        error_msg = f"Failed to initialize LLM {request.model_name} for provider {request.llm_provider}."
        log_message(error_msg, "ERROR")
        return GenerateTestCasesResponse(error=error_msg)
    if not embeddings: # Embeddings are critical for RAG in generate_test_cases
        error_msg = f"Failed to initialize Embeddings for provider {request.llm_provider} (or fallback). RAG cannot proceed."
        log_message(error_msg, "ERROR")
        return GenerateTestCasesResponse(error=error_msg)

    # 3. Call the core test case generation logic
    try:
        generation_results = llm_core.generate_test_cases(
            text=request.main_requirements_text,
            selected_apps=request.selected_applications,
            uploaded_context_files_content=request.application_contexts, # Pass the dict of content strings
            llm=llm,
            embeddings=embeddings,
            provider_name=request.llm_provider
        )
        log_message(f"Test case generation results: {generation_results}", "INFO")
        return GenerateTestCasesResponse(results=generation_results)
    except Exception as e:
        error_msg = f"An unexpected error occurred during test case generation: {str(e)}"
        log_message(error_msg, "ERROR", exc_info=True)
        return GenerateTestCasesResponse(error=error_msg)

@app.post("/api/actions/refactor/single", response_model=RefactorSingleTestCaseResponse)
async def refactor_single_test_case_action(request: RefactorSingleTestCaseRequest):
    """
    Refactors a single test case based on user instructions.
    """
    log_message(f"Refactor single TC request for app: {request.app_name}, TC ID: {request.tc_id}", "INFO")

    # 1. Check credentials (RAG not strictly needed for refactor, so require_fallback_for_rag=False)
    creds_ok, creds_msg = llm_core.check_credentials(
        provider=request.llm_provider,
        credentials=request.api_credentials,
        fallback_key=request.openai_fallback_api_key or "",
        require_fallback_for_rag=False
    )
    if not creds_ok:
        log_message(f"Credential check failed for refactor_single_tc: {creds_msg}", "WARNING")
        return RefactorSingleTestCaseResponse(error=f"Credential check failed: {creds_msg}")

    # 2. Initialize LLM (embeddings not needed for this refactor function)
    llm, _ = llm_core.get_llm_and_embeddings(
        provider=request.llm_provider,
        model_name=request.model_name,
        credentials=request.api_credentials,
        fallback_openai_key=request.openai_fallback_api_key or ""
    )

    if not llm:
        error_msg = f"Failed to initialize LLM {request.model_name} for provider {request.llm_provider}."
        log_message(error_msg, "ERROR")
        return RefactorSingleTestCaseResponse(error=error_msg)

    # 3. Call the core refactoring logic
    try:
        refactored_data = llm_core.refactor_single_test_case(
            app_name=request.app_name,
            tc_id=request.tc_id,
            instructions=request.instructions,
            original_tc_data=request.original_tc_data,
            llm=llm,
            provider_name=request.llm_provider
        )

        if refactored_data:
            log_message(f"Single TC refactor successful for TC ID: {request.tc_id}", "INFO")
            return RefactorSingleTestCaseResponse(refactored_tc_data=refactored_data)
        else:
            # This case implies the llm_core function returned None, meaning an internal error or bad LLM output
            error_msg = "Refactoring failed: LLM did not return valid data or an internal error occurred."
            log_message(error_msg, "WARNING") # Warning because the request was processed but no valid data came back
            return RefactorSingleTestCaseResponse(error=error_msg)

    except Exception as e:
        error_msg = f"An unexpected error occurred during single test case refactoring: {str(e)}"
        log_message(error_msg, "ERROR", exc_info=True)
        return RefactorSingleTestCaseResponse(error=error_msg)

@app.post("/api/actions/refactor/all/{app_name}", response_model=RefactorAllTestCasesResponse)
async def refactor_all_test_cases_action(app_name: str, request: RefactorAllTestCasesRequest):
    """
    Refactors all test cases for a given application based on user instructions.
    """
    log_message(f"Refactor all TCs request for app: {app_name}", "INFO")

    # 1. Check credentials
    creds_ok, creds_msg = llm_core.check_credentials(
        provider=request.llm_provider,
        credentials=request.api_credentials,
        fallback_key=request.openai_fallback_api_key or "",
        require_fallback_for_rag=False
    )
    if not creds_ok:
        log_message(f"Credential check failed for refactor_all_tcs: {creds_msg}", "WARNING")
        return RefactorAllTestCasesResponse(app_name=app_name, error=f"Credential check failed: {creds_msg}")

    # 2. Initialize LLM
    llm, _ = llm_core.get_llm_and_embeddings(
        provider=request.llm_provider,
        model_name=request.model_name,
        credentials=request.api_credentials,
        fallback_openai_key=request.openai_fallback_api_key or ""
    )

    if not llm:
        error_msg = f"Failed to initialize LLM {request.model_name} for provider {request.llm_provider}."
        log_message(error_msg, "ERROR")
        return RefactorAllTestCasesResponse(app_name=app_name, error=error_msg)

    # 3. Call the core bulk refactoring logic
    try:
        refactored_list = llm_core.refactor_all_test_cases(
            app_name=app_name, # Pass app_name from path
            instructions=request.instructions,
            original_tc_list=request.original_tc_list,
            llm=llm,
            provider_name=request.llm_provider
        )

        if refactored_list is not None: # Could be an empty list on success
            log_message(f"Bulk TC refactor successful for app: {app_name}", "INFO")
            return RefactorAllTestCasesResponse(app_name=app_name, refactored_test_cases=refactored_list)
        else:
            error_msg = "Bulk refactoring failed: LLM did not return valid data or an internal error occurred."
            log_message(error_msg, "WARNING")
            return RefactorAllTestCasesResponse(app_name=app_name, error=error_msg)

    except Exception as e:
        error_msg = f"An unexpected error occurred during bulk test case refactoring: {str(e)}"
        log_message(error_msg, "ERROR", exc_info=True)
        return RefactorAllTestCasesResponse(app_name=app_name, error=error_msg)

@app.post("/api/actions/review/{app_name}", response_model=AIReviewResponse)
async def ai_review_action(app_name: str, request: AIReviewRequest):
    """
    Performs AI review of existing test cases for a given application.
    """
    log_message(f"AI Review request for app: {app_name}, provider: {request.llm_provider}", "INFO")

    # 1. Check credentials (RAG not strictly needed for review, so require_fallback_for_rag=False)
    creds_ok, creds_msg = llm_core.check_credentials(
        provider=request.llm_provider,
        credentials=request.api_credentials,
        fallback_key=request.openai_fallback_api_key or "",
        require_fallback_for_rag=False 
    )
    if not creds_ok:
        log_message(f"Credential check failed for ai_review: {creds_msg}", "WARNING")
        return AIReviewResponse(app_name=app_name, error=f"Credential check failed: {creds_msg}")

    # 2. Initialize LLM (embeddings not strictly needed by perform_ai_test_case_review itself, but get_llm_and_embeddings returns it)
    llm, _ = llm_core.get_llm_and_embeddings(
        provider=request.llm_provider,
        model_name=request.model_name,
        credentials=request.api_credentials,
        fallback_openai_key=request.openai_fallback_api_key or ""
    )

    if not llm:
        error_msg = f"Failed to initialize LLM {request.model_name} for provider {request.llm_provider}."
        log_message(error_msg, "ERROR")
        return AIReviewResponse(app_name=app_name, error=error_msg)

    # 3. Call the core AI review logic
    try:
        review_data = llm_core.perform_ai_test_case_review(
            main_requirements_text=request.main_requirements_text,
            additional_context_str=request.additional_context_str,
            existing_test_cases=request.existing_test_cases,
            llm=llm,
            provider_name=request.llm_provider
        )

        if review_data:
            log_message(f"AI Review successful for app: {app_name}", "INFO")
            return AIReviewResponse(app_name=app_name, review_results=review_data)
        else:
            error_msg = "AI Review failed: LLM did not return valid data or an internal error occurred."
            log_message(error_msg, "WARNING")
            return AIReviewResponse(app_name=app_name, error=error_msg)

    except Exception as e:
        error_msg = f"An unexpected error occurred during AI review: {str(e)}"
        log_message(error_msg, "ERROR", exc_info=True)
        return AIReviewResponse(app_name=app_name, error=error_msg)

@app.post("/api/actions/review/apply/{app_name}", response_model=ApplyAIReviewResponse)
async def apply_ai_review_action(app_name: str, request: ApplyAIReviewRequest):
    """
    Applies user decisions from an AI review to the set of test cases for an application.
    """
    log_message(f"Apply AI Review changes request for app: {app_name}", "INFO")

    try:
        updated_tcs, summary = llm_core.apply_ai_review_changes_logic(
            app_name=app_name,
            existing_test_cases=request.existing_test_cases,
            ai_review_suggestions_processed=request.ai_review_suggestions_processed,
            user_decisions=request.user_decisions
        )
        return ApplyAIReviewResponse(
            app_name=app_name,
            updated_test_cases=updated_tcs,
            summary_message=summary
        )
    except Exception as e:
        error_msg = f"An unexpected error occurred while applying AI review changes: {str(e)}"
        log_message(error_msg, "ERROR", exc_info=True)
        return ApplyAIReviewResponse(app_name=app_name, updated_test_cases=[], summary_message="", error=error_msg)

@app.post("/api/export/excel")
async def export_excel(request: ExportRequest):
    """
    Exports the provided test cases to an Excel file.
    The request body should contain the test cases data and an optional filename.
    """
    log_message(f"Excel export request received for filename: {request.filename}", "INFO")
    try:
        # Filter out any app entries where the result is an error string instead of a list of TCs
        valid_test_cases_data = {
            app: tcs for app, tcs in request.test_cases_data.items() if isinstance(tcs, list)
        }

        if not valid_test_cases_data:
            log_message("Excel export failed: No valid test case lists provided.", "WARNING")
            raise HTTPException(status_code=400, detail="No valid test case data provided for export.")

        excel_bytes_io = export_test_cases_to_excel_bytes(valid_test_cases_data)
        
        if excel_bytes_io is None:
            log_message("Excel export failed: Could not generate Excel file.", "ERROR")
            raise HTTPException(status_code=500, detail="Failed to generate Excel file.")
            
        # Ensure filename is safe and has .xlsx extension
        filename = request.filename if request.filename.endswith(".xlsx") else f"{request.filename}.xlsx"
        # Basic sanitization for filename (more robust might be needed for production)
        filename = "".join(c if c.isalnum() or c in ('.', '-', '_') else '_' for c in filename)
        
        log_message(f"Successfully generated excel bytes for {filename}", "INFO")
        
        return StreamingResponse(
            excel_bytes_io,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except HTTPException:
        raise # Re-raise HTTPException from above
    except Exception as e:
        log_message(f"Error during Excel export: {e}", "ERROR", exc_info=True)
        # Return a JSON error response instead of raising HTTPException for non-HTTP specific errors
        # This allows frontend to potentially parse it, though for file download, it's tricky.
        # For simplicity here, we'll raise HTTPException, but a real app might handle this differently.
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during Excel export: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
