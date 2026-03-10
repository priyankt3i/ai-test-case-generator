# """Stores configuration constants and settings for the application."""

import os

# --- Application Settings ---
APP_TITLE = "📄 TestGen AI 🧪"
PAGE_LAYOUT = "wide"
ACCEPTED_FILE_TYPES = ["docx", "pdf"]
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

Important to Remember: Customer Interface is also called CI or eSales. So make sure you do not duplicate. Prefer the Name - eSales in Place of CI or Customer Interface.

Important to Remember: Agent Interface is also called AI or Agent Web (portal). So make sure you do not duplicate. Prefer the Name - AgentWeb in Place of AI or Agent Interface.

If no applications can be clearly identified, return an empty list: []

Requirements Text:
```{text}```

Identified Applications (Python list format only):
"""

GENERATE_TC_PROMPT_TEMPLATE = """You are an expert QA Analyst generating test cases based on provided requirements context. Create detailed, actionable test cases.
**Carefully consider both the 'Requirements Context Retrieved' below and each test case should be mapped to FR or BR ID, for traceability AND any 'Additional Context' provided within the 'User Input Query/Focus' when generating the test cases, especially for populating fields like 'Test Data'.**

Format your response *only* as a single JSON list of objects. Each object represents a test case and must include these fields: `{field_names}`.
The 'Test Steps' and 'Expected Results' should be lists of strings, where each string is a separate step or result. The number of test steps should match the number of expected results.
Populate `source_chunk_id` with the chunk identifier(s) used from the retrieved context (for example: `CHUNK_0003` or `CHUNK_0003,CHUNK_0007`).
Populate `source_requirement_excerpt` with a short direct excerpt from the retrieved requirement context that justifies the test case.

Ensure the JSON is valid. Do not include any text before or after the JSON list.

Requirements Context Retrieved:
```{{context}}```

User Input Query/Focus:
```{{input}}```

JSON Output (List of Test Case Objects):
"""

REFACTOR_TC_PROMPT_TEMPLATE = """You are an expert QA Analyst modifying an existing test case based on user instructions and retrieved requirement context.

Return *only* the complete, updated JSON object for the *single* test case being modified. Ensure all original fields are present unless the instructions specifically dictate removal.
The `Test Case ID` should generally remain `{tc_id}`, unless explicitly asked to change it. Ensure the output is a valid JSON object, with no surrounding text.
You must keep or update traceability fields:
- `source_chunk_id`: chunk identifier(s) used from requirement context.
- `source_requirement_excerpt`: short requirement excerpt that justifies the updated test case.

Requirements Context Retrieved:
```text
{requirements_context_retrieved}
```

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
REFACTOR_ALL_TC_PROMPT_TEMPLATE = """You are an expert QA Analyst modifying a list of existing test cases based on general user instructions and retrieved requirement context. Apply the instructions thoughtfully to each test case in the provided list.

Return *only* a complete JSON list containing the updated JSON objects for *all* the test cases provided.
- Ensure each object in the returned list is a valid JSON object representing a test case.
- Preserve the original `Test Case ID` for each test case unless the instructions specifically ask to change IDs across the board.
- Ensure all original fields are present in each updated test case object unless the instructions specifically dictate removal or modification.
- The number of test cases in the output list should match the number in the input list.
- Every test case must include traceability fields:
  - `source_chunk_id`
  - `source_requirement_excerpt`

Requirements Context Retrieved:
```text
{requirements_context_retrieved}
```

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

# --- NEW: AI Review Test Cases Prompt ---
AI_REVIEW_TC_PROMPT_TEMPLATE = """You are an expert QA Lead tasked with reviewing a set of test cases against business requirements and supplementary context.
Your goal is to identify coverage gaps, suggest improvements to existing test cases, recommend new test cases, and flag duplicates.

**Inputs Provided to You:**
1.  `requirements_context_retrieved`: Requirement chunks retrieved from vector search for this review.
2.  `additional_context`: Concatenated text from all supplementary context documents.
3.  `existing_test_cases`: A JSON list of current test cases. Each test case object has the following fields: {field_names}.

**Your Task:**
Analyze the `existing_test_cases` in light of the `requirements_context_retrieved` and `additional_context`.
Return a *single JSON object* with the following top-level keys:
-   `coverage_summary`: (String) A brief summary of how well the existing test cases cover the requirements.
-   `newly_suggested_test_cases`: (JSON List of Objects) A list of new test case objects you recommend. Each object *must* conform to the fields: {field_names}, including `source_chunk_id` and `source_requirement_excerpt`. If no new test cases are needed, provide an empty list [].
-   `modified_test_cases_suggestions`: (JSON List of Objects) A list of suggestions for modifying existing test cases. Each object in this list *must* have:
    -   `original_test_case_id`: (String) The 'Test Case ID' of the test case to be modified.
    -   `modification_reason`: (String) A brief explanation of why the modification is suggested.
    -   `suggested_test_case_data`: (JSON Object) The complete test case data for the *modified* version, including all fields: {field_names}, and include `source_chunk_id` and `source_requirement_excerpt`.
    If no modifications are needed, provide an empty list [].
-   `identified_duplicates`: (JSON List of Objects) A list of objects, where each object represents a group of duplicate or highly redundant test cases. Each object *must* have:
    -   `duplicate_group_id`: (String) A unique identifier for this group of duplicates (e.g., "DUP_GROUP_1").
    -   `test_case_ids`: (JSON List of Strings) A list of 'Test Case ID's that are considered duplicates of each other.
    -   `reason`: (String) Why these are considered duplicates.
    If no duplicates are found, provide an empty list [].

**Important Formatting Rules:**
-   The entire output *must* be a single valid JSON object.
-   All test case objects (new or suggested modifications) *must* include all the fields: {field_names}.
-   Ensure all string values within the JSON are properly escaped.
-   Do not include any text, explanations, or apologies outside of the main JSON object.

**Inputs:**

Requirements Context Retrieved (`requirements_context_retrieved`):
```text
{{requirements_context_retrieved}}
```

Additional Context (`additional_context`):
```text
{{additional_context}}
```

Existing Test Cases (`existing_test_cases` - JSON list):
```json
{{existing_test_cases_json}}
```

**Your JSON Output Only:**
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
        "notes": "Requires OpenAI API Key.",
        "pricing": {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }
    },
    "Gemini": {
        "models": ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-pro", "gemini-2.5-pro", "gemini-2.5-flash"],
        "credentials": ["api_key"],
        "llm_module": "langchain_google_genai",
        "llm_class": "ChatGoogleGenerativeAI",
        "embeddings_module": "langchain_google_genai",
        "embeddings_class": "GoogleGenerativeAIEmbeddings",
        "embeddings_model_id": "models/gemini-embedding-001",
        "embeddings_model_ids": [
            "models/gemini-embedding-001",
            "gemini-embedding-001",
            "models/text-embedding-004",
            "text-embedding-004",
            "models/embedding-001",
            "embedding-001"
        ],
        "notes": "Requires Google API Key (often called GOOGLE_API_KEY).",
        "pricing": {
            "gemini-1.5-pro-latest": {"input": 0.0035, "output": 0.0105},
            "gemini-1.5-flash-latest": {"input": 0.00035, "output": 0.00105},
            "gemini-pro": {"input": 0.00025, "output": 0.0005},
            "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
            "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025}
        }
    },
    "Claude": {
        "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "credentials": ["api_key"],
        "llm_module": "langchain_anthropic",
        "llm_class": "ChatAnthropic",
        "embeddings_module": None,
        "embeddings_class": None,
        "notes": "Requires Anthropic API Key. **RAG embedding uses OpenAI fallback.**",
        "pricing": {
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125}
        }
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
        "models": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
        "credentials": ["api_key"],
        "llm_module": "langchain_groq",
        "llm_class": "ChatGroq",
        "embeddings_module": None,
        "embeddings_class": None,
        "notes": "Requires Groq API Key. **RAG embedding uses OpenAI fallback.**",
        "pricing": {
            "llama3-8b-8192": {"input": 0.00005, "output": 0.00008},
            "llama3-70b-8192": {"input": 0.00059, "output": 0.00079},
            "mixtral-8x7b-32768": {"input": 0.00024, "output": 0.00024},
            "gemma-7b-it": {"input": 0.00007, "output": 0.00007}
        }
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
EXCEL_EXPECTED_COLUMNS = [
    'Test Case ID',
    'FR/BR ID',
    'Test Case Name',
    'Description',
    'Preconditions',
    'Step #',
    'Test Steps',
    'Expected Results',
    'Test Data',
    'Priority',
    'source_requirement_excerpt',
    'source_chunk_id'
]
EXCEL_MAX_COL_WIDTH = 60
EXCEL_DEFAULT_COL_WIDTH = 20
EXCEL_SHEET_NAME_MAX_LEN = 31

# --- Note: Default prompt templates are defined above ---
