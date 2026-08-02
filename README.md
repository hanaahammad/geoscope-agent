# GeoScope Agent

GeoScope is an AOI-aware GeoAI assistant for remote-sensing dataset selection and workflow planning.

It combines an interactive map, STAC satellite metadata, a local document knowledge base, vector retrieval, local LLM generation, retrieval evaluation, LLM-as-a-judge, logging, monitoring, and user feedback in one Streamlit application.

## Main use cases

GeoScope can support questions related to:

- Agriculture and crop monitoring
- Sentinel-1 SAR analysis
- Sentinel-2 optical analysis
- Landsat and urban heat
- MODIS and phenology
- Flood assessment
- Land-cover change
- GeoAI and remote-sensing foundation models

## What makes GeoScope different?

GeoScope does not return a generic answer only from an LLM. It combines the user question, selected Area of Interest, application context, available Sentinel-2 scene metadata, and relevant technical guidance retrieved from the knowledge base.

## Core concepts

### Remote sensing

Remote sensing is the observation of the Earth without direct physical contact. Satellites record reflected or emitted energy from vegetation, water, land, buildings, and the atmosphere.

### AOI

AOI means Area of Interest. It is the polygon or rectangle selected by the user on the map and included in the application context and logs.

### STAC

STAC means SpatioTemporal Asset Catalog. GeoScope uses a STAC API to search Sentinel-2 scenes by geographic area, date range, cloud-cover threshold, and collection.

The current MVP retrieves metadata and preview links. It does not download or process full satellite images.

### RAG

RAG means Retrieval-Augmented Generation. Before the LLM answers, GeoScope retrieves relevant document chunks from the local vector database and provides them as context.

## Main workflow

```text
Prepare documents
      ↓
Build vector index
      ↓
Draw AOI
      ↓
Search Sentinel-2 scenes through STAC
      ↓
Ask a GeoAI question
      ↓
Retrieve relevant document chunks
      ↓
Generate a grounded answer
      ↓
Evaluate retrieval and generation
      ↓
Log and monitor the run
```

## Application pages

### 1. Data Preparation

- Upload PDF, HTML, or HTM files
- Add documents to `docs/raw`
- Run document ingestion
- Extract and chunk text
- Build or update the Chroma vector index
- Review document-processing results

### 2. AOI and STAC

- Draw an AOI on an interactive map
- Select a date range
- Define a maximum cloud-cover threshold
- Search Sentinel-2 scenes
- Review scene metadata and previews
- Save the AOI and STAC context for the next pages

### 3. Ask GeoAI

- Select an application
- Select a crop and season
- Choose from example questions
- Ask a custom question
- Retrieve relevant document chunks
- Generate a local LLM answer
- Review the retrieved sources

### 4. Evaluation and Feedback

Retrieval evaluation includes ground-truth questions, expected documents, top-k search, Hit Rate, Mean Reciprocal Rank, and failed retrieval cases.

Generation evaluation includes LLM-as-a-judge, relevance, groundedness, completeness, technical correctness, citation quality, geographic relevance, overall score, and judge comments.

### 5. Monitoring

- Logged questions
- Application, crop, and season
- AOI description
- STAC scene count
- Date range
- Cloud-cover threshold
- Latency
- Run status
- Interactive charts
- Search and filters
- CSV export

## Architecture

```text
Streamlit UI
    │
    ├── Document upload and ingestion
    │       └── PDF / HTML / HTM
    │
    ├── Chroma vector store
    │       └── Ollama embeddings
    │
    ├── AOI map
    │       └── Folium / streamlit-folium
    │
    ├── STAC search
    │       └── Earth Search
    │
    ├── RAG generation
    │       └── Ollama local model
    │
    ├── Evaluation
    │       ├── Hit Rate and MRR
    │       └── LLM-as-a-judge
    │
    └── Monitoring
            ├── DuckDB
            └── dlt
```

## Technology stack

- Python
- Streamlit
- Folium
- streamlit-folium
- ChromaDB
- Ollama
- DuckDB
- dlt
- pandas
- requests
- PyPDF
- BeautifulSoup
- Altair

## Local models

Recommended local configuration:

- Generator: `qwen2.5:7b-instruct`
- Embeddings: `nomic-embed-text:latest`
- Judge: `llama3.1:8b`
- Optional reasoning judge: `deepseek-r1:8b`

## Project structure

```text
GeoScope_Agent/
├── GeoScope.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── pages/
│   ├── 1_Data_Preparation.py
│   ├── 2_AOI_and_STAC.py
│   ├── 3_Ask_GeoAI.py
│   ├── 4_Evaluation_and_Feedback.py
│   └── 5_Monitoring.py
├── src/
│   ├── __init__.py
│   ├── ui.py
│   ├── ingest_documents.py
│   ├── build_vector_index.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── evaluation.py
│   ├── stac_search.py
│   ├── monitoring.py
│   └── dlt_logging.py
├── docs/
│   ├── raw/
│   └── processed/
├── data/
│   ├── evaluation_questions.csv
│   └── vector_store/
└── logs/
```

## Installation

### 1. Clone or download the project

```powershell
git clone <YOUR_REPOSITORY_URL>
cd GeoScope_Agent
```

### 2. Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install and start Ollama

```powershell
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

Optional:

```powershell
ollama pull deepseek-r1:8b
```

## Prepare the knowledge base

Place supported documents in `docs/raw`, or upload them from the Data Preparation page.

Then run ingestion and build the vector index from the Streamlit interface, or run:

```powershell
python src\ingest_documents.py
python src\build_vector_index.py
```

## Run the application

```powershell
python -m streamlit run GeoScope.py
```

## Evaluation dataset

The retrieval evaluation dataset is stored in `data/evaluation_questions.csv`.

Each question contains a question ID, question, expected document, domain, and difficulty.

Retrieval quality is measured using Hit Rate and Mean Reciprocal Rank.

## Monitoring and logging

GeoScope stores application runs in DuckDB, including the question, answer, retrieved sources, application, crop, season, AOI summary and geometry, STAC scene count, date range, cloud-cover threshold, latency, status, and errors.

Evaluation and feedback records can also be loaded through dlt into DuckDB.

## Current limitations

- The MVP does not download full satellite scenes.
- The STAC workflow currently focuses on Sentinel-2.
- The local LLM may be slow depending on hardware.
- Uploaded documents must be ingested and re-indexed before retrieval.
- JavaScript-heavy HTML pages may not extract correctly.
- GeoScope supports expert review but does not replace scientific validation.

## Future improvements

- OpenAI access for reviewers
- Model-provider selector
- Secure API-key configuration
- Sentinel-1 STAC support
- More AOI-aware scene-ranking logic
- Online deployment
- Authentication
- Richer experiment tracking
- More real user questions in the evaluation dataset
- Agent and tool-trajectory evaluation

## Reviewer mode

A future reviewer mode will allow users to choose between local Ollama models and OpenAI models.

The API key will be provided securely through Streamlit secrets or environment variables and will never be stored in the repository.

## Responsible use

GeoScope recommendations should be reviewed by a qualified remote-sensing or domain specialist before operational or policy use.

The application is designed for research, learning, prototyping, and decision support.

## Author

Hanaa Hammad  
LLM Zoomcamp Capstone Project
