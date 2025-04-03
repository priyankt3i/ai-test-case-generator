# 📄 Business Requirements to Test Cases Generator 🧪

## Overview

This Streamlit application leverages the power of Large Language Models (LLMs) via LangChain to automate the often tedious process of generating functional test cases from business requirement documents. Users can upload a `.docx` file, configure their preferred LLM provider, identify applications within the document, generate detailed test cases using Retrieval-Augmented Generation (RAG), optionally enhance generation with specific application context from YAML files, refactor generated test cases, and export the results to Excel.

## ✨ Features

* **📄 Document Upload:** Accepts business requirements in `.docx` format.
* **🤖 Multi-LLM Support:** Integrates with various LLM providers:
    * OpenAI (GPT-4o, GPT-4 Turbo, etc.)
    * Google Gemini (Gemini 1.5 Pro/Flash, etc.)
    * Anthropic Claude 3 (Opus, Sonnet, Haiku) - *Requires OpenAI API Key for RAG embeddings fallback.*
    * AWS Bedrock (various models like Claude, Titan, Llama3 via AWS)
    * Groq (Llama3, Mixtral, Gemma via Groq API) - *Requires OpenAI API Key for RAG embeddings fallback.*
* **🔑 Secure Configuration:** API keys and credentials managed via the UI sidebar (using password inputs where appropriate).
* **🎯 Application Identification:** Automatically identifies distinct applications or systems mentioned within the requirements document using an LLM.
* **🧠 RAG Test Case Generation:** Employs Retrieval-Augmented Generation (RAG) to generate relevant test cases:
    * Splits the input document into chunks.
    * Creates vector embeddings using the selected provider's model (or OpenAI fallback).
    * Retrieves relevant document sections based on the target application.
    * Uses an LLM chain (`create_stuff_documents_chain`) to generate test cases based on retrieved context and user query.
* **🧩 Optional YAML Context:** Enhance test case generation by providing additional context (e.g., API details, data models, existing test snippets) via `.yaml` files placed in an `app_context` folder. Context can be selected per identified application.
* **✍️ Test Case Refactoring:** Modify and refine individual generated test cases using LLM instructions via a dedicated UI tab.
* **📊 Results Display:** Presents generated test cases clearly in tables within expandable sections per application. Includes summary metrics.
* **📋 Session Logging:** Provides a dedicated "Session Logs" tab acting as a UI console, displaying debug messages and errors for the current session. Includes "Clear Log" and "Copy Log" functionality.
* **💾 Excel Export:** Exports the generated (and potentially refactored) test cases to a well-formatted `.xlsx` file, with separate sheets per application.

## 🛠️ Technologies Used

* **Frontend:** Streamlit
* **LLM Orchestration:** `LangChain` & `LangChain Community/Core`
* **LLM Providers:** `langchain-openai`, `langchain-google-genai`, `langchain-anthropic`, `langchain-aws`, `langchain-groq`
* **Vector Store:** FAISS (`faiss-cpu`)
* **Document Parsing:** `python-mammoth`
* **Data Handling:** `Pandas`
* **Configuration:** `PyYAML`

## 🖼️ Screenshots

![Main Interface](./public/main.jpg)

## 📁 Project Structure

```txt

ai-test_case_generator/
├── main_app.py         # Main Streamlit script, UI orchestration
├── config.py           # Configuration (LLM providers, prompts, constants)
├── utils.py            # Utility functions (logging, imports, sanitization)
├── file_processing.py  # .docx text extraction
├── llm_integration.py  # LangChain/LLM logic (init, identify, generate, refactor)
├── excel_export.py     # Excel file generation logic
├── ui_components.py    # Functions for rendering UI parts (sidebar, results)
├── app_context/        # Optional: Folder for user-provided YAML context files
│   ├── AppName1.yaml
│   └── AppName2.yml
└── requirements.txt    # Python dependencies

```

## ⚙️ Setup Instructions

1.  **Prerequisites:**
    * Python (version 3.11.9 or higher recommended). Test on 3.12.3
    * `pip` (Python package installer).
    * Git (optional, for cloning).

2.  **Clone or Download:**
    * Clone the repository: `git clone https://github.com/priyankt3i/ai-test-case-generator.git`
    * OR Download the source code files (`main_app.py`, `config.py`, etc.) into a single project directory.

3.  **Create Virtual Environment (Recommended):**
    * Navigate to the project directory in your terminal.
    * Create the environment: `python -m venv .venv` (or `py -m venv .venv`)
    * Activate the environment:
        * Windows: `.\.venv\Scripts\activate`
        * macOS/Linux: `source .venv/bin/activate`

4.  **Install Dependencies:**
    * Ensure your virtual environment is active (you should see `(.venv)` in your terminal prompt).
    * Create a `requirements.txt` file (see [Dependencies](#dependencies) below for an example).
    * Install the required packages: `pip install -r requirements.txt`

5.  **Prepare API Keys:** Gather the necessary API keys for the LLM providers you intend to use (OpenAI, Google AI Studio, Anthropic, AWS, Groq). Note that some providers (Claude, Groq) require an OpenAI key *in addition* to their own key if you want to use the RAG generation feature, as they use OpenAI for fallback embeddings in this implementation.

6.  **Create Context Folder (Optional):**
    * If you want to provide additional context for specific applications, create a folder named `app_context` inside your main project directory (the same directory where `main_app.py` resides).
    * Place your context files inside this folder, naming them appropriately (e.g., `MyApplication.yaml`, `AnotherSystem.yml`). The application will detect these files based on their base names.

## ▶️ Running the Application

1.  **Activate Virtual Environment:** If not already active, navigate to the project directory in your terminal and activate the environment (`.\.venv\Scripts\activate` or `source .venv/bin/activate`).
2.  **Run Streamlit:** Execute the following command:
    ```bash
    streamlit run main_app.py
    ```
3.  **Access App:** Streamlit will provide a local URL (usually `http://localhost:8501` or similar). Open this URL in your web browser.

## 🖱️ Using the Application

1.  **Sidebar Configuration:**
    * **Upload Document:** Use the file uploader to select your `.docx` requirements file.
    * **LLM Configuration:** Choose your desired LLM Provider (e.g., OpenAI) and the specific Model from the dropdowns.
    * **API Credentials:** Enter the required API key(s) for the selected provider. For providers needing fallback embeddings (Claude, Groq), also enter the OpenAI API key in the designated field if you plan to use the "Generate Test Cases" feature. Credentials are treated as passwords where applicable.
    * **Optional App Context:** This section shows the status of the `app_context` folder.

2.  **Generate Test Cases Tab:**
    * **(Step 1: Upload)** Implicitly done via the sidebar.
    * **(Step 2: Identify Applications)** Click the "Identify Applications" button. The app will use the configured LLM to list applications found in the document.
    * **(Step 3: Select Apps & Context)**
        * Select the applications you want to generate test cases for from the multi-select box.
        * For each selected application, choose an optional `.yaml` context file from the corresponding dropdown (files are detected from the `app_context` folder). Select "None" if no specific context file applies.
    * **(Step 4: Generate Test Cases)** Click the "Generate Cases for X Application(s)" button. The app will perform RAG for each selected application, using the main document and any chosen YAML context. Progress is shown.
    * **(Results)** Generated test cases (or errors) will appear in expandable sections below the summary metrics.

3.  **Refactor Test Cases Tab:**
    * This tab allows you to modify existing test cases generated in the current session.
    * Select the **Application** and the **Test Case ID** you want to modify from the dropdowns. The original test case data will be shown for reference.
    * Enter clear **Modification Instructions** in the text area (e.g., "Change priority to High", "Add step to verify login").
    * Click "Get Refactored Version". The LLM will attempt the modification.
    * A **Confirmation UI** will appear showing the Original and Proposed versions side-by-side.
    * Click "✅ Apply Change" to update the test case in the current session's results or "❌ Discard Change" to ignore the proposed modification.

4.  **Session Logs Tab:**
    * Displays timestamped debug messages and error details generated during the current application session.
    * Useful for troubleshooting issues with API calls, RAG setup, or file processing.
    * **Clear Log:** Removes all messages from the current log view.
    * **Copy Log:** Copies the entire log content to your clipboard.

5.  **Export Results:**
    * Once test cases have been generated (and potentially refactored), an "Export Results" section appears below the tabs.
    * Click the "Download All Test Cases (.xlsx)" button to save the results to an Excel file. Each application's test cases will be on a separate sheet.

## 🔧 Configuration Details

* **`config.py`:** Contains core settings like LLM provider details, prompt templates, RAG parameters (chunk size, overlap), and Excel export settings. Modify this file for advanced configuration.
* **`app_context/` Folder:** Place `.yaml` or `.yml` files here for application-specific context. The application matches the base filename (without extension) to the identified application names during context selection.

## 📦 Dependencies

Ensure you have a `requirements.txt` file. Based on the libraries used, it should look something like this (adjust versions as needed or based on your `pip freeze`):

```txt
aiohappyeyeballs==2.6.1
aiohttp==3.11.14
aiosignal==1.3.2
altair==5.5.0
annotated-types==0.7.0   
anyio==4.9.0
attrs==25.3.0
blinker==1.9.0
cachetools==5.5.2        
certifi==2025.1.31       
charset-normalizer==3.4.1
click==8.1.8
cobble==0.1.4
colorama==0.4.6
dataclasses-json==0.6.7  
distro==1.9.0
faiss-cpu==1.10.0
frozenlist==1.5.0
gitdb==4.0.12
GitPython==3.1.44
greenlet==3.1.1
h11==0.14.0
httpcore==1.0.7
httpx==0.28.1
httpx-sse==0.4.0
idna==3.10
Jinja2==3.1.6
jiter==0.9.0
jsonpatch==1.33
jsonpointer==3.0.0
jsonschema==4.23.0
jsonschema-specifications==2024.10.1
langchain==0.3.22
langchain-community==0.3.20
langchain-core==0.3.49
langchain-openai==0.3.11
langchain-text-splitters==0.3.7
langsmith==0.3.20
mammoth==1.9.0
MarkupSafe==3.0.2
marshmallow==3.26.1
multidict==6.2.0
mypy-extensions==1.0.0
narwhals==1.33.0
numpy==2.2.4
openai==1.70.0
orjson==3.10.16
packaging==24.2
pandas==2.2.3
pillow==11.1.0
propcache==0.3.1
protobuf==5.29.4
pyarrow==19.0.1
pydantic==2.11.1
pydantic-settings==2.8.1
pydantic_core==2.33.0
pydeck==0.9.1
python-dateutil==2.9.0.post0
python-dotenv==1.1.0
pytz==2025.2
PyYAML==6.0.2
referencing==0.36.2
regex==2024.11.6
requests==2.32.3
requests-toolbelt==1.0.0
rpds-py==0.24.0
six==1.17.0
smmap==5.0.2
sniffio==1.3.1
SQLAlchemy==2.0.40
streamlit==1.44.0
tenacity==9.0.0
tiktoken==0.9.0
toml==0.10.2
tornado==6.4.2
tqdm==4.67.1
typing-inspect==0.9.0
typing-inspection==0.4.0
typing_extensions==4.13.0
tzdata==2025.2
urllib3==2.3.0
watchdog==6.0.0
XlsxWriter==3.2.2
yarl==1.18.3
zstandard==0.23.0
