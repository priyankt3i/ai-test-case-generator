# TestGen AI

Streamlit application that converts requirements into test cases using LLMs and RAG.

## What It Does
- Upload one or more requirements documents (`.docx`, `.pdf`).
- Identify applications/systems mentioned in requirements.
- Generate test cases per selected application using RAG over the requirements corpus.
- Review generated test cases with AI (coverage, new suggestions, modifications, duplicates).
- Refactor test cases (single or bulk) with instruction-driven updates.
- Export final test cases to Excel.

## Key Implementation Notes
- RAG index uses FAISS over requirement chunks (`CHUNK_SIZE`, `CHUNK_OVERLAP` in `config.py`).
- Generate, AI review, and refactor flows are RAG-grounded.
- Requirement vectorstores are cached in Streamlit session state and reused across generate/review/refactor when source text + embedding signature match.
- Output schema includes explicit traceability fields:
  - `source_requirement_excerpt`
  - `source_chunk_id`

## Project Structure
```txt
ai-test-case-generator/
|-- main_app.py
|-- llm_integration_core.py
|-- config.py
|-- ui_components.py
|-- helper/
|   |-- utils.py
|   |-- file_processing.py
|   `-- excel_export.py
|-- llm_providers/
|   |-- llm_openai.py
|   |-- llm_gemini.py
|   |-- llm_claude.py
|   |-- llm_bedrock.py
|   |-- llm_groq.py
|   |-- llm_ollama.py
|   `-- llm_embeddings_utils.py
|-- app_context/
|-- public/
`-- requirements.txt
```

## Main Flows
1. Identify Applications
- Uses selected provider/model to extract application names from uploaded requirements text.

2. Generate Test Cases (RAG)
- Builds or reuses cached requirement vectorstore.
- Retrieves relevant chunks per selected app.
- Generates JSON test cases with required schema and traceability fields.

3. AI Review Test Cases (RAG)
- Retrieves requirement chunks relevant to current app/test-case set.
- Produces coverage summary, new test cases, modification suggestions, and duplicate groups.
- Suggested/new cases include traceability fields.

4. Manual Refactor (RAG)
- Single and bulk refactor retrieve requirement chunks first.
- Refactor outputs preserve schema and Test Case IDs, and include traceability fields.

5. Export
- Exports all app results to `.xlsx` with expected columns from `config.EXCEL_EXPECTED_COLUMNS`.

## Supported Providers
Configured in `config.py`:
- OpenAI
- Gemini
- Claude (embedding fallback via OpenAI)
- AWS Bedrock
- Groq (embedding fallback via OpenAI)
- Ollama

## Run
```bash
pip install -r requirements.txt
streamlit run main_app.py
```

## Configuration
Use `config.py` for:
- Prompt templates
- Provider/model lists
- RAG settings (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVER_SEARCH_K`)
- Excel schema (`EXCEL_EXPECTED_COLUMNS`)