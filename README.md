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
* **LLM Orchestration:** LangChain & LangChain Community/Core
* **LLM Providers:** `langchain-openai`, `langchain-google-genai`, `langchain-anthropic`, `langchain-aws`, `langchain-groq`
* **Vector Store:** FAISS (`faiss-cpu`)
* **Document Parsing:** `python-mammoth`
* **Data Handling:** Pandas
* **Configuration:** PyYAML
* **UI Components:** `streamlit-clipboard`
* **Language:** Python 3

## 🖼️ Screenshots

*(Placeholder: You can add screenshots of the application UI here)*

* *Screenshot of the main interface with sidebar.*
* *Screenshot showing identified applications and context selection.*
* *Screenshot of the generated results table.*
* *Screenshot of the refactoring tab.*
* *Screenshot of the session logs tab.*

## 📁 Project Structure

test_case_generator/├── main_app.py         # Main Streamlit script, UI orchestration├── config.py           # Configuration (LLM providers, prompts, constants)├── utils.py            # Utility functions (logging, imports, sanitization)├── file_processing.py  # .docx text extraction├── llm_integration.py  # LangChain/LLM logic (init, identify, generate, refactor)├── excel_export.py     # Excel file generation logic├── ui_components.py    # Functions for rendering UI parts (sidebar, results)├── app_context/        # Optional: Folder for user-provided YAML context files│   ├── AppName1.yaml│   └── AppName2.yml└── requirements.txt    # Python dependencies
## ⚙️ Setup Instructions

1.  **Prerequisites:**
    * Python (version 3.9 or higher recommended).
    * `pip` (Python package installer).
    * Git (optional, for cloning).

2.  **Clone or Download:**
    * Clone the repository: `git clone <repository_url>`
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
# Core Framework
streamlit>=1.44.0
streamlit-clipboard>=0.0.6

# LLM Orchestration & Core
langchain>=
