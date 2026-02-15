# 🚀 Aether-Agent: Autonomous Aerospace Research Agent

An autonomous AI agent system for technical synthesis and orbital mechanics research, designed for TU Delft Master's portfolio.

## Overview

Aether-Agent is an intelligent research assistant that automatically discovers, processes, and reasons over aerospace research papers. It combines automated paper acquisition, semantic knowledge ingestion, and agentic reasoning with strict anti-hallucination guardrails.

## Core Components

### 1. **Automated Library** (`research_fetcher.py`)
- Automatically downloads aerospace research papers from arXiv
- Searches for papers on:
  - Large Language Models for Aerospace Engineering
  - Satellite Image Segmentation for Forest Fire Detection
  - Hohmann Transfer Trajectory Optimization for CubeSats
  - Small Satellite Thermal Management Systems
- Prevents duplicate downloads
- Maintains `library_manifest.json` with paper metadata

### 2. **Grounded Knowledge** (`ingest.py`)
- Processes PDFs using `PyMuPDFLoader` for high-fidelity extraction
- Performs semantic chunking with `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap)
- Stores embeddings in ChromaDB using `all-mpnet-base-v2` model
- Creates a searchable knowledge base for technical queries

### 3. **Agentic Reasoning** (`agent_core.py`)
Implements a ReAct (Reasoning + Acting) loop with three specialized tools:

- **Search_Library**: Semantic search in ChromaDB knowledge base
- **Paper_Finder**: Automatically fetches new papers from arXiv when knowledge is insufficient
- **Math_Engine**: Python REPL for executing orbital mechanics calculations

### 4. **Interactive UI** (`app.py`)
- Streamlit-based web interface
- Real-time thought process visualization
- Citation tracking and display
- Conversation history

### 5. **Verification Suite** (`eval_suite.py`)
- Uses RAGAS framework for evaluation
- Scores **Faithfulness** (grounding in sources)
- Scores **Relevance** (answer quality)
- Generates evaluation reports

## Technical Standards

### Anti-Hallucination Guardrails
- **"Thought" must precede "Action"**: All reasoning steps are logged
- **Mandatory Citations**: Every claim includes Paper ID and page number
- **Uncertainty Handling**: Returns "Technical data currently unavailable" when no data found
- **Physics over Chat**: Mathematical calculations executed in Python REPL, not estimated

## Installation

1. **Clone or download the project**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Quick Setup** (recommended):
```bash
python setup.py
```
This will automatically fetch papers and ingest them into the knowledge base.

**OR** Manual setup:

3a. **Download research papers**:
```bash
python research_fetcher.py
```

3b. **Ingest papers into knowledge base**:
```bash
python ingest.py
```

## Usage

### Command Line Interface

Test the agent directly:
```bash
python agent_core.py
```

### Web Interface

Launch the Streamlit app:
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Evaluation

Run the evaluation suite:
```bash
python eval_suite.py
```

## Project Structure

```
Space_LLM/
├── research_papers/          # Auto-populated PDF library
├── chroma_db/               # Vector database storage
├── library_manifest.json     # Paper metadata tracking
├── research_fetcher.py      # Paper acquisition module
├── ingest.py                # Knowledge ingestion pipeline
├── agent_core.py            # ReAct agent implementation
├── app.py                   # Streamlit UI
├── eval_suite.py            # RAGAS evaluation framework
├── setup.py                 # Quick setup script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Example Queries

- "What are the key challenges in thermal management for small satellites?"
- "How do you calculate delta-v for a Hohmann transfer?"
- "What are Large Language Models used for in aerospace engineering?"
- "How is satellite image segmentation used for forest fire detection?"

## Key Features

✅ **Autonomous Paper Discovery**: Automatically finds and downloads relevant research  
✅ **Semantic Search**: High-quality embeddings for technical content retrieval  
✅ **Self-Improving**: Fetches new papers when knowledge is insufficient  
✅ **Mathematical Accuracy**: Executes physics calculations in Python  
✅ **Citation Tracking**: Every answer includes source citations  
✅ **Thought Transparency**: Full reasoning trace visible in UI  
✅ **Evaluation Framework**: RAGAS-based scoring for faithfulness and relevance  

## Technical Stack

- **PDF Processing**: PyMuPDF
- **Vector Database**: ChromaDB
- **Embeddings**: sentence-transformers/all-mpnet-base-v2
- **Text Processing**: LangChain
- **Paper Source**: arXiv API
- **UI Framework**: Streamlit
- **Evaluation**: RAGAS

## License

This project is designed for academic portfolio purposes.

## Author

Developed for TU Delft Master's portfolio - Aerospace Research Agent project.

---

**Note**: Ensure you have sufficient disk space for PDF downloads and ChromaDB storage. The system will automatically manage duplicates and maintain a clean knowledge base.
