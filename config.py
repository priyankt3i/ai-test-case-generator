# config.py
"""Stores configuration constants and settings for the application."""

import os

# --- Application Settings ---
APP_TITLE = "📄 Business Requirements to Test Cases Generator 🧪"
PAGE_LAYOUT = "wide"
ACCEPTED_FILE_TYPES = ["docx"]
APP_CONTEXT_FOLDER_NAME = "app_context" # Folder name relative to script execution dir
NO_CONTEXT_OPTION = "None" # Display text for selecting no context file
DEFAULT_TEMPERATURE = 0.0 # Default LLM temperature for deterministic output

# Construct the absolute path for the context folder based on execution directory
APP_CONTEXT_FOLDER_PATH = os.path.join(os.getcwd(), APP_CONTEXT_FOLDER_NAME)


# --- LLM Provider Configuration ---
# (Config remains the same - omitted for brevity)
LLM_PROVIDER_CONFIG = {
    "OpenAI": {
        "models": ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "credentials": ["api_key"],
        "llm_module": "langchain_openai",
        "llm_class": "ChatOpenAI",
        "embeddings_module": "langchain_openai",
        "embeddings_class": "OpenAIEmbeddings",
        "notes": "Requires OpenAI API Key."
    },
    "Gemini": {
        "models": ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-pro"],
        "credentials": ["api_key"],
        "llm_module": "langchain_google_genai",
        "llm_class": "ChatGoogleGenerativeAI",
        "embeddings_module": "langchain_google_genai",
        "embeddings_class": "GoogleGenerativeAIEmbeddings",
        "embeddings_model_id": "models/embedding-001", # Specific embedding model for Gemini
        "notes": "Requires Google API Key (often called GOOGLE_API_KEY)."
    },
    "Claude": {
        "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "credentials": ["api_key"],
        "llm_module": "langchain_anthropic",
        "llm_class": "ChatAnthropic",
        "embeddings_module": None, # No dedicated LangChain embedding class
        "embeddings_class": None,
        "notes": "Requires Anthropic API Key. **RAG embedding uses OpenAI fallback (requires OpenAI API key below).**"
    },
    "AWS Bedrock": {
        "models": [
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "amazon.titan-text-express-v1",
            "cohere.command-r-v1:0",
            "meta.llama3-8b-instruct-v1:0"
        ],
        "embedding_model_ids": [
            "amazon.titan-embed-text-v1",
            "cohere.embed-english-v3",
            "cohere.embed-multilingual-v3"
        ],
        "credentials": ["aws_access_key_id", "aws_secret_access_key", "region_name", "embedding_model_id"],
        "llm_module": "langchain_aws",
        "llm_class": "ChatBedrock",
        "embeddings_module": "langchain_aws",
        "embeddings_class": "BedrockEmbeddings",
        "notes": "Requires AWS Access Key ID, Secret Access Key, and Region Name. Select a compatible Embedding Model ID enabled in your account."
    },
     "Groq": {
        "models": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
        "credentials": ["api_key"],
        "llm_module": "langchain_groq",
        "llm_class": "ChatGroq",
        "embeddings_module": None,
        "embeddings_class": None,
        "notes": "Requires Groq API Key. **RAG embedding uses OpenAI fallback (requires OpenAI API key below).**"
    },
    # *** NEW OLLAMA ENTRY ***
    "Ollama": {
        "models": ["llama3", "mistral", "phi3", "codellama", "gemma"], # Common models, user needs them pulled
        "credentials": ["base_url"], # Base URL for server, model selected from list
        "llm_module": "langchain_community.chat_models.ollama", # Path to ChatOllama
        "llm_class": "ChatOllama",
        "embeddings_module": "langchain_community.embeddings.ollama", # Path to OllamaEmbeddings
        "embeddings_class": "OllamaEmbeddings",
        "notes": "Requires Ollama server running locally. Ensure the selected model is pulled (`ollama pull <model_name>`). Default Base URL is http://localhost:11434."
    }
    # *** END NEW OLLAMA ENTRY ***
}
FALLBACK_EMBEDDING_PROVIDERS = ["Claude", "Groq"]

# --- RAG Settings ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVER_SEARCH_K = 5

# --- Excel Export Settings ---
EXCEL_EXPORT_FILENAME = "generated_test_cases.xlsx"
EXCEL_EXPECTED_COLUMNS = ['Test Case ID', 'Test Case Name', 'Description', 'Preconditions', 'Test Steps', 'Expected Results', 'Test Data', 'Priority']
EXCEL_MAX_COL_WIDTH = 60
EXCEL_DEFAULT_COL_WIDTH = 20
EXCEL_SHEET_NAME_MAX_LEN = 31

# --- Prompt Templates ---
IDENTIFY_APP_PROMPT_TEMPLATE = """You are a senior QA analyst tasked with identifying the primary software applications or systems being discussed in a set of business requirements. Focus on distinct applications, not features within an application unless they are presented as separate systems.

Return *only* a Python-style list of strings, where each string is an identified application name. Do not include explanations, apologies, or any text outside the list.

Example: ["App One", "System Two", "Reporting Module"]

If no applications can be clearly identified, return an empty list: []

Requirements Text:
```{text}```

Identified Applications (Python list format only):
"""

# *** UPDATED TEMPLATE: Added sentence to emphasize using additional context ***
GENERATE_TC_PROMPT_TEMPLATE = """You are an expert QA Analyst generating test cases based on provided requirements context. Create detailed, actionable test cases.
**Carefully consider both the 'Requirements Context Retrieved' below AND any 'Additional Context' provided within the 'User Input Query/Focus' when generating the test cases, especially for populating fields like 'Test Data'.**

Format your response *only* as a single JSON list of objects. Each object represents a test case and must include these fields: `{field_names}`.
Ensure the JSON is valid. Do not include any text before or after the JSON list.

Requirements Context Retrieved:
```{{context}}```

User Input Query/Focus:
```{{input}}```

JSON Output (List of Test Case Objects):
"""

REFACTOR_TC_PROMPT_TEMPLATE = """You are an expert QA Analyst modifying an existing test case based on user instructions.

Return *only* the complete, updated JSON object for the *single* test case being modified. Ensure all original fields are present unless the instructions specifically dictate removal.
The `Test Case ID` should generally remain `{tc_id}`, unless explicitly asked to change it. Ensure the output is a valid JSON object, with no surrounding text.

Original Test Case JSON:
```json
{original_tc_json}
```

User Modification Instructions:
```
{user_instructions}
```

Updated Test Case JSON Object Only:
"""
