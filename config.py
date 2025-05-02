# """Stores configuration constants and settings for the application."""

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

# --- Default Prompt Templates ---
# (Keep your existing default templates here as fallbacks)
IDENTIFY_APP_PROMPT_TEMPLATE = """You are a senior QA analyst tasked with identifying the primary software applications or systems being discussed in a set of business requirements. Focus on distinct applications, not features within an application unless they are presented as separate systems.

Return *only* a Python-style list of strings, where each string is an identified application name. Do not include explanations, apologies, or any text outside the list.

Example: ["App One", "System Two", "Reporting Module"]

If no applications can be clearly identified, return an empty list: []

Requirements Text:
```{text}```

Identified Applications (Python list format only):
"""

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

# --- NEW: Bulk Refactoring Prompt ---
REFACTOR_ALL_TC_PROMPT_TEMPLATE = """You are an expert QA Analyst modifying a list of existing test cases based on general user instructions. Apply the instructions thoughtfully to each test case in the provided list.

Return *only* a complete JSON list containing the updated JSON objects for *all* the test cases provided.
- Ensure each object in the returned list is a valid JSON object representing a test case.
- Preserve the original `Test Case ID` for each test case unless the instructions specifically ask to change IDs across the board.
- Ensure all original fields are present in each updated test case object unless the instructions specifically dictate removal or modification.
- The number of test cases in the output list should match the number in the input list.

Original Test Case List (JSON):
```json
{original_tc_list_json}
```

User Modification Instructions (Apply to ALL test cases):
```
{user_instructions}
```

Updated Test Case List (JSON Array Only):
"""


# --- LLM Provider Configuration ---
LLM_PROVIDER_CONFIG = {
    "OpenAI": {
        "models": ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "credentials": ["api_key"],
        "llm_module": "langchain_openai",
        "llm_class": "ChatOpenAI",
        "embeddings_module": "langchain_openai",
        "embeddings_class": "OpenAIEmbeddings",
        "notes": "Requires OpenAI API Key."
        # No 'prompt_templates' needed if defaults work well for OpenAI
    },
    "Gemini": {
        "models": ["gemini-2.5-pro-preview-03-25", 
                   "gemini-2.5-flash-preview-04-17",
                   "gemini-2.5.pro-exp-03-25", 
                   "gemini-1.5-flash-latest", 
                   "gemini-1.5-pro-latest", 
                   "gemini-pro"],
        "credentials": ["api_key"],
        "llm_module": "langchain_google_genai",
        "llm_class": "ChatGoogleGenerativeAI",
        "embeddings_module": "langchain_google_genai",
        "embeddings_class": "GoogleGenerativeAIEmbeddings",
        "embeddings_model_id": "models/embedding-001",
        "notes": "Requires Google API Key (often called GOOGLE_API_KEY)."
    },
    "Claude": {
        "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "credentials": ["api_key"],
        "llm_module": "langchain_anthropic",
        "llm_class": "ChatAnthropic",
        "embeddings_module": None,
        "embeddings_class": None,
        "notes": "Requires Anthropic API Key. **RAG embedding uses OpenAI fallback.**"
    },
    "AWS Bedrock": {
        "models": [
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
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
        "credentials": ["aws_access_key_id", "aws_secret_access_key", "aws_session_token", "region_name", "embedding_model_id"],
        "llm_module": "langchain_aws",
        "llm_class": "ChatBedrock",
        "embeddings_module": "langchain_aws",
        "embeddings_class": "BedrockEmbeddings",
        "notes": "Requires AWS Credentials and Region. Select Embedding Model ID."
    },
     "Groq": {
        "models": ["deepseek-r1-distill-qwen-32b", "deepseek-r1-distill-llama-70b", "llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
        "credentials": ["api_key"],
        "llm_module": "langchain_groq",
        "llm_class": "ChatGroq",
        "embeddings_module": None,
        "embeddings_class": None,
        "notes": "Requires Groq API Key. **RAG embedding uses OpenAI fallback.**"
    },
    # *** MODIFIED OLLAMA ENTRY ***
    "Ollama": {
        "models": ["llama3", "mistral", "phi3", "codellama", "gemma"],
        "credentials": ["base_url"],
        "llm_module": "langchain_ollama", # Use new package if available
        "llm_class": "ChatOllama",
        "embeddings_module": "langchain_ollama", # Use new package if available
        "embeddings_class": "OllamaEmbeddings",
        "notes": "Requires Ollama server running. Ensure model is pulled.",
        # *** ADDED prompt_templates dictionary for overrides ***
        "prompt_templates": {
            "IDENTIFY_APP": """You are an Senior Software QA Expert. Your task is to extract ONLY the names of software applications or distinct software systems from the following text.
            IGNORE features, file paths, URLs, or code snippets.

            Focus ONLY on high-level known application names like 'eSales' (Also known as CI or Customer Interface), 'AgenetWeb' (Also known as AI or Agenet Interface),  'Quick Quote', 'Polstar' (Backend SQLServer DB Legacy), 'PolicyPro' (Backend Oracle DB New built on RDS AWS), 'Billing System'.

            Output ONLY a Python list of strings containing these names.

            Example Output: ["App One", "System Two"]

            If no application names are found, output: []

            Do NOT include any other text, explanation, or apologies.

            Text to analyze:
            ```{text}```

            Python List Output:""",
            # You could add overrides for "GENERATE_TC" or "REFACTOR_TC" here too if needed
            # "GENERATE_TC": """Ollama-specific generation prompt...""",
        }
        # *** END prompt_templates ***
    }
    # *** END MODIFIED OLLAMA ENTRY ***
}

# --- Fallback and RAG settings ---
FALLBACK_EMBEDDING_PROVIDERS = ["Claude", "Groq"]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVER_SEARCH_K = 5

# --- Excel Export Settings ---
EXCEL_EXPORT_FILENAME = "generated_test_cases.xlsx"
EXCEL_EXPECTED_COLUMNS = ['Test Case ID', 'Test Case Name', 'Description', 'Preconditions', 'Test Steps', 'Expected Results', 'Test Data', 'Priority']
EXCEL_MAX_COL_WIDTH = 60
EXCEL_DEFAULT_COL_WIDTH = 20
EXCEL_SHEET_NAME_MAX_LEN = 31

# --- Note: Default prompt templates are defined above ---
