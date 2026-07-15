# Agentic RAG Research Assistant for Document AI

This repository contains a Colab-ready notebook that builds an end-to-end **agentic retrieval-augmented generation research assistant** for technical question answering over local PDFs and recent arXiv papers in Document AI.

The main notebook is:

```text
Agentic_RAG_Research_Assistant_V1.ipynb
```

It combines layout-aware PDF ingestion, arXiv corpus expansion, hybrid retrieval, LangGraph orchestration, local LLM synthesis, contradiction detection, claim-level grounding checks, and interactive demo interfaces.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Runtime Configuration](#runtime-configuration)
- [Notebook Workflow](#notebook-workflow)
- [Outputs](#outputs)
- [Evaluation](#evaluation)
- [Demo Interfaces](#demo-interfaces)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

The project is designed as a research-oriented assistant for answering questions such as:

```text
What are the latest methods for document layout analysis?
```

Instead of answering from a fixed prompt alone, the assistant builds a searchable corpus from:

- User-provided PDF files
- arXiv papers fetched from a configurable search query
- Layout-preserved text chunks extracted from PDF pages

The assistant then retrieves supporting evidence, synthesizes a grounded research answer, checks source contradictions, and verifies generated claims with an asynchronous hallucination guard.

## Key Features

- **Layout-aware PDF parsing** using PyMuPDF block extraction.
- **arXiv ingestion** for building a fresh research corpus.
- **Hybrid retrieval** with dense semantic search and BM25 keyword search.
- **Reciprocal Rank Fusion** to merge retrieval signals.
- **LangGraph workflow** for agentic retrieval, synthesis, contradiction checks, and guardrail routing.
- **Local LLM answer synthesis** with `microsoft/Phi-3-mini-4k-instruct`.
- **Grounding verification** with `cross-encoder/nli-deberta-v3-small`.
- **Claim-level support labels**: `SUPPORTED`, `PARTIALLY_SUPPORTED`, and `UNVERIFIED`.
- **Automated evaluation** over 20 document-AI research questions.
- **Interactive demos** through Streamlit, Gradio, and notebook widgets.
- **Visual diagnostics** for retrieval scores, latency, grounding, and evaluation metrics.

## System Architecture

```text
Local PDFs + arXiv Papers
          |
          v
Layout-Aware PDF Extraction
          |
          v
Chunking + Metadata Preservation
          |
          v
Dense Embeddings + BM25 Index
          |
          v
Hybrid Retrieval + Reciprocal Rank Fusion
          |
          v
LangGraph Research Assistant
          |
          +--> Contradiction Detection
          |
          +--> Phi-3 Research Synthesis
          |
          +--> Async NLI Grounding Guard
          |
          v
Grounded Research Answer + Sources + Metrics
```

## Repository Structure

```text
.
|-- Agentic_RAG_Research_Assistant_V1.ipynb       # Main notebook
|-- Agentic_RAG_Research_Assistant_Colab.ipynb    # Earlier Colab variant
|-- make_agentic_rag_notebook.py                  # Notebook generation helper
|-- requirements.txt                              # Existing project dependencies
|-- app.py                                        # Separate Streamlit demo from the broader repo
|-- mediform_core.py                              # Separate document-form utilities
`-- README.md                                     # Project documentation
```

During execution, the main notebook writes these generated files inside the active runtime:

```text
research_assistant_core.py     # Reusable assistant implementation
research_streamlit_app.py      # Streamlit demo application
```

## Quick Start

### 1. Open the Notebook

Open `Agentic_RAG_Research_Assistant_V1.ipynb` in Google Colab.

A GPU runtime is recommended:

```text
Runtime > Change runtime type > T4 GPU
```

### 2. Install Dependencies

The notebook includes its own Colab setup cell:

```bash
pip uninstall -y torchvision torchaudio
pip install -U langchain langchain-community langgraph \
  faiss-cpu sentence-transformers pymupdf streamlit requests arxiv \
  "transformers>=4.44" accelerate sentencepiece
```

### 3. Add PDFs

Upload local PDFs into the configured folder:

```text
/content/pdfs
```

The notebook can also fetch papers from arXiv, so local PDFs are optional for a quick demo.

### 4. Run the Notebook

Run the cells from top to bottom. The notebook will:

1. Configure the runtime.
2. Write the reusable assistant module.
3. Build the local PDF and arXiv corpus.
4. Create retrieval indexes.
5. Initialize the LangGraph assistant.
6. Run an example research query.
7. Display retrieved sources and grounding checks.
8. Evaluate the assistant.
9. Launch interactive demo interfaces.

## Runtime Configuration

The main configuration cell exposes the most important runtime controls:

```python
PDF_FOLDER = "/content/pdfs"
ARXIV_QUERY = "document layout understanding"
ARXIV_MAX_RESULTS = 20

LOAD_LLM = True
LOAD_NLI = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

| Setting | Purpose |
| --- | --- |
| `PDF_FOLDER` | Folder containing local PDF files to index. |
| `ARXIV_QUERY` | Search query used to fetch papers from arXiv. |
| `ARXIV_MAX_RESULTS` | Maximum number of arXiv papers to add to the corpus. |
| `LOAD_LLM` | Loads Phi-3-mini for full synthesis when `True`; uses a lightweight fallback when `False`. |
| `LOAD_NLI` | Enables NLI-based contradiction and grounding checks. |
| `DEVICE` | Uses CUDA when available, otherwise CPU. |

For a fast CPU-only smoke test, use:

```python
LOAD_LLM = False
LOAD_NLI = False
```

## Notebook Workflow

The notebook is organized into the following sections:

1. **Install Dependencies**  
   Installs LangChain, LangGraph, FAISS, Sentence Transformers, PyMuPDF, Streamlit, arXiv, Transformers, and model acceleration packages.

2. **Configure the Runtime**  
   Sets corpus paths, arXiv query parameters, model-loading flags, and device selection.

3. **Write the Core Research Assistant Module**  
   Creates `research_assistant_core.py`, which contains the reusable implementation for ingestion, indexing, retrieval, graph orchestration, synthesis, guardrails, and evaluation.

4. **Build the Corpus and Indexes**  
   Loads local PDFs, fetches arXiv papers, chunks documents, embeds text, and builds FAISS and BM25 indexes.

5. **Create the LangGraph Assistant**  
   Instantiates the retrieval and reasoning workflow.

6. **Run an Example Query**  
   Executes a sample research question and prints a grounded answer.

7. **Inspect Sources and Grounding**  
   Displays retrieved passages, rank scores, source metadata, and claim-level verification results.

8. **Evaluate the Assistant**  
   Runs a 20-question evaluation harness and summarizes retrieval, faithfulness, guardrail, and latency metrics.

9. **Create and Launch Demo Apps**  
   Writes a Streamlit app and provides Gradio and notebook-widget alternatives.

10. **Visualize Performance**  
    Generates plots for retrieval scoring, grounding confidence, claim precision, support rate, and latency.

## Outputs

After a successful run, the notebook produces:

- A searchable PDF and arXiv research corpus.
- Dense and sparse retrieval indexes.
- Grounded research answers with inline citations.
- Retrieved-source tables.
- Claim-level grounding tables.
- Contradiction summaries.
- Evaluation results across the built-in question set.
- Retrieval breakdown plots.
- Latency and guardrail dashboards.
- A generated Streamlit app for interactive use.
- A Gradio interface for notebook-friendly demos.

## Evaluation

The evaluation harness reports:

- Faithfulness score
- Claim precision
- Claim support rate
- Retry trigger rate
- Time to first answer
- Synthesis latency
- Guardrail latency
- Total latency

The goal is not only to produce plausible answers, but to measure whether generated claims are supported by retrieved evidence.

## Demo Interfaces

### Streamlit

The notebook writes a Streamlit app:

```text
research_streamlit_app.py
```

In Colab, the notebook launches it with:

```bash
streamlit run research_streamlit_app.py --server.port 8501 --server.headless true
```

The app supports:

- PDF upload
- arXiv search configuration
- Index building
- Research chat
- Retrieved-source inspection
- Guard confidence display
- Exportable research notes

### Gradio

The notebook also creates a Gradio interface for lightweight interactive querying directly inside the notebook.

### Notebook Widgets

If Streamlit tunneling is unavailable, the notebook includes a direct query fallback that runs the same assistant without leaving Colab.

## Troubleshooting

### Phi-3 Loading Errors

Use `transformers>=4.44`. The notebook intentionally avoids old Phi-3 remote-code loading paths that can trigger `rope_scaling` configuration errors.

### Slow Runtime

Use a GPU runtime when `LOAD_LLM = True`. For a quick validation run, set:

```python
LOAD_LLM = False
LOAD_NLI = False
ARXIV_MAX_RESULTS = 3
```

### Streamlit Does Not Open in Colab

Use the notebook's direct query fallback or the Gradio interface. Both exercise the same assistant logic.

### No Local PDFs

The assistant can still run by fetching arXiv papers. Local PDFs are useful for custom corpora, but they are not required for the default demo path.

## License

MIT License
