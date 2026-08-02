# GeoScope Agent

**An AOI-aware GeoAI assistant for remote-sensing dataset selection and analysis planning**

GeoScope Agent is an end-to-end LLM application built for the LLM Zoomcamp capstone project. It helps users choose suitable Earth-observation datasets and processing workflows based on a selected **Area of Interest (AOI)**, application, crop, season, and natural-language question.

The project combines:

- an interactive map for AOI selection;
- a curated catalogue of remote-sensing datasets;
- document retrieval from GeoAI and satellite documentation;
- a local LLM served through Ollama;
- retrieval and generation evaluation;
- user feedback collection;
- application monitoring in Streamlit.

> **Current status:** the starter MVP already supports AOI drawing, dataset filtering, local Ollama generation, DuckDB logging, and a recent-runs view. Document ingestion, vector retrieval, evaluation, and feedback are the next implementation steps.

---

## 1. Problem Statement

Remote-sensing users often need to answer questions such as:

- Which satellite dataset is suitable for my area?
- Which dataset is appropriate for wheat monitoring?
- Should I use optical, thermal, or radar imagery?
- Which spatial and temporal resolution is required?
- What preprocessing and analysis steps should I follow?
- Are there curated datasets for a specific region or crop?

Generic recommendations are often insufficient because the correct answer depends on:

- the selected geographic area;
- the intended application;
- crop type and season;
- spatial and temporal resolution;
- cloud conditions;
- locally available curated data.

GeoScope Agent narrows the answer using the AOI and user-defined filters, then combines structured dataset information with retrieved knowledge documents to produce a practical recommendation.

---

## 2. Main Use Case

A user draws an AOI on the map, selects an application such as **Crop monitoring**, chooses a crop such as **Wheat**, and asks:

> Which datasets and workflow should I use to monitor wheat in this selected area?

GeoScope Agent then:

1. captures the AOI geometry;
2. filters the curated dataset catalogue;
3. retrieves relevant GeoAI documentation;
4. builds a grounded prompt;
5. generates a recommendation with a local LLM;
6. evaluates retrieval and answer quality;
7. logs the run and user feedback;
8. displays monitoring indicators in Streamlit.

---

## 3. Course Requirements Mapping

| LLM Zoomcamp requirement | GeoScope implementation |
|---|---|
| Select a dataset or API-backed data source | Curated remote-sensing dataset catalogue and GeoAI knowledge documents |
| Ingest data into a knowledge base | Python ingestion pipeline, chunking, embeddings, and vector storage |
| Implement the application flow | AOI filtering, retrieval, prompt construction, local LLM generation, and optional tool calls |
| Evaluate the RAG or agent flow | Retrieval metrics and LLM-as-a-judge generation evaluation |
| Create an interface | Streamlit application with an interactive AOI map |
| Collect user feedback | Thumbs up/down and optional comments |
| Monitor the application | DuckDB logs and a Streamlit monitoring dashboard |

---

## 4. Architecture

```text
Remote-sensing documents      Curated dataset catalogue
           |                            |
           v                            v
   Ingestion and chunking        Structured filtering
           |                            |
           v                            |
      Vector store                     |
           \                           /
            \                         /
             v                       v
          AOI-aware retrieval and tools
                       |
                       v
               Prompt construction
                       |
                       v
              Ollama local LLM
                       |
                       v
          Answer, sources, limitations
                       |
              +--------+--------+
              |                 |
              v                 v
        Evaluation         User feedback
              \                 /
               \               /
                v             v
                DuckDB monitoring
                       |
                       v
              Streamlit dashboard
```

---

## 5. Planned Application Flow

```text
Select AOI
   |
Choose application, crop, and season
   |
Ask a question
   |
Filter structured dataset catalogue
   |
Retrieve relevant document chunks
   |
Build grounded prompt
   |
Generate answer with Ollama
   |
Evaluate retrieval and generation
   |
Collect feedback and log the run
```

A simple LangGraph workflow may be added after the complete RAG flow works:

```text
parse_question
   -> validate_aoi
   -> retrieve_documents
   -> search_dataset_catalogue
   -> generate_answer
   -> evaluate_answer
   -> log_run
```

LangGraph is optional for the capstone requirements. It is used only if it improves clarity and orchestration.

---

## 6. Data Sources

### 6.1 Structured dataset catalogue

The project includes a curated CSV catalogue with examples such as:

- Sentinel-1 GRD;
- Sentinel-2 Level-2A;
- Landsat 8/9 Surface Reflectance;
- MODIS vegetation products;
- ECOSTRESS land-surface temperature;
- curated regional crop calendars.

Important catalogue fields include:

```text
dataset_id
dataset_name
provider
applications
supported_crops
countries
regions
spatial_resolution
temporal_resolution
season
sensor
data_type
cloud_sensitive
access_method
documentation_source
curation_status
```

### 6.2 Knowledge documents

The knowledge base will contain a small, carefully selected collection of:

- NASA ARSET training materials;
- Sentinel product documentation;
- Landsat guides;
- ECOSTRESS documentation;
- crop-monitoring workflows;
- GeoAI and remote-sensing reference material;
- selected project notes.

Documents should be stored in `data/raw/` with metadata identifying the source, topic, sensor, application, region, and publication date.

---

## 7. Technology Stack

| Technology | Role |
|---|---|
| Python | Main application and pipeline language |
| Streamlit | User interface and monitoring dashboard |
| Folium / streamlit-folium | Interactive AOI map and polygon drawing |
| Ollama | Local hosting of generation, embedding, and judge models |
| Qwen 2.5 7B Instruct | Main answer-generation model |
| Nomic Embed Text | Document embeddings |
| Llama 3.1 8B | LLM-as-a-judge evaluation |
| DuckDB | Structured catalogue queries, logs, metrics, and feedback |
| Chroma or Qdrant | Vector storage and semantic retrieval |
| LangGraph | Optional orchestration of the application flow |

The project intentionally avoids heavy infrastructure such as Airflow, Kestra, Neo4j, Kubernetes, and complex multi-agent autonomy in the first capstone.

Neo4j is reserved for a possible second capstone focused on GeoAI knowledge graphs and lineage.

---

## 8. Local Models

The default local models are:

```text
Generation: qwen2.5:7b-instruct
Embeddings: nomic-embed-text:latest
Evaluation: llama3.1:8b
```

Optional models already available locally may be used for experiments:

```text
deepseek-r1:8b
llava:7b
qwen2.5-coder:7b
llama3:latest
```

LLaVA may be used later for map or image description, but it is not treated as a scientific remote-sensing foundation model.

---

## 9. Repository Structure

```text
GeoScope_Agent/
|
|-- app/
|   |-- main.py
|   |-- pages/
|   |   |-- 1_Ask_GeoAI.py
|   |   |-- 2_Evaluation.py
|   |   `-- 3_Monitoring.py
|   `-- components/
|
|-- src/
|   |-- ingestion.py
|   |-- retrieval.py
|   |-- generation.py
|   |-- evaluation.py
|   |-- monitoring.py
|   `-- feedback.py
|
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- dataset_catalog.csv
|   `-- evaluation_questions.csv
|
|-- logs/
|-- screenshots/
|-- docs/
|   |-- setup.md
|   |-- usage.md
|   |-- architecture.md
|   `-- evaluation.md
|
|-- requirements.txt
|-- .env.example
|-- .gitignore
`-- README.md
```

The starter version currently contains a smaller subset of this structure. Additional folders will be introduced as the corresponding features are implemented.

---

## 10. Setup

### 10.1 Prerequisites

- Windows 10 or 11
- Python 3.11
- Ollama installed and running
- At least one supported local generation model

### 10.2 Clone or open the project

```bat
cd C:\Workspaces\LLM-ZOOMCAMP\GeoScope_Agent
```

### 10.3 Create a virtual environment

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 10.4 Install dependencies

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 10.5 Check Ollama models

```bat
ollama list
```

Pull the required models when necessary:

```bat
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

### 10.6 Start Ollama

```bat
ollama serve
```

If Ollama is already running, the command may report that the port is in use. In that case, continue to the next step.

### 10.7 Run the application

```bat
streamlit run app\main.py
```

---

## 11. Configuration

A future `.env.example` file will contain:

```text
OLLAMA_BASE_URL=http://localhost:11434
GENERATION_MODEL=qwen2.5:7b-instruct
EMBEDDING_MODEL=nomic-embed-text:latest
JUDGE_MODEL=llama3.1:8b
VECTOR_DB_PATH=data/vector_store
LOG_DB_PATH=logs/runs.duckdb
```

No paid API key is required for the default local configuration.

---

## 12. How to Use the Current MVP

1. Start the Streamlit application.
2. Draw a rectangle or polygon on the map.
3. Choose an application.
4. Choose a crop and season.
5. Enter a question.
6. Review the filtered dataset catalogue.
7. Click **Run GeoScope**.
8. Read the generated recommendation.
9. Review recent runs at the bottom of the page.

Example input:

```text
Application: Crop monitoring
Crop: Wheat
Season: Winter
Question: Which datasets and workflow should I use to monitor wheat in this AOI?
```

Expected answer structure:

- direct recommendation;
- primary dataset;
- complementary dataset;
- processing workflow;
- limitations;
- indication when the catalogue is insufficient.

---

## 13. Example Use Cases

### Crop monitoring

```text
Which datasets should I use to monitor wheat in this AOI?
```

### Cloud-prone agricultural area

```text
How should I combine Sentinel-1 and Sentinel-2 for crop monitoring in this area?
```

### Urban heat

```text
Which datasets are suitable for urban heat analysis in Cairo?
```

### Land-cover change

```text
Which dataset and workflow should I use to detect land-cover change between two years?
```

### Flood assessment

```text
Which radar dataset is appropriate for flood mapping in the selected AOI?
```

---

## 14. Evaluation

The project evaluates both retrieval and generation.

### 14.1 Retrieval evaluation

Planned metrics:

- Hit Rate;
- Mean Reciprocal Rank;
- Precision@k;
- expected document retrieved;
- expected dataset retrieved;
- AOI and metadata-filter correctness.

The evaluation dataset will contain approximately 20 to 30 questions with:

```text
question
expected_document
expected_dataset
expected_topic
reference_answer
important_terms
```

### 14.2 Generation evaluation

The LLM judge will score:

- relevance;
- groundedness;
- completeness;
- geographic relevance;
- crop relevance;
- technical correctness;
- citation quality;
- uncertainty disclosure.

Example output:

```json
{
  "relevance": 5,
  "groundedness": 4,
  "geographic_relevance": 5,
  "technical_correctness": 4,
  "citation_quality": 4,
  "overall_score": 4.4
}
```

Automated evaluation does not replace expert validation. Remote-sensing recommendations should still be reviewed by a knowledgeable user.

---

## 15. Feedback Collection

The final application will allow users to provide:

- thumbs up;
- thumbs down;
- optional written comment;
- optional reason for negative feedback.

Feedback will be stored with the corresponding run ID in DuckDB.

Planned fields:

```text
run_id
feedback_value
feedback_comment
created_at
```

---

## 16. Monitoring

Each run will record:

- timestamp;
- question;
- AOI geometry;
- application, crop, and season;
- retrieved documents;
- matched datasets;
- generated answer;
- retrieval metrics;
- judge scores;
- latency;
- status and errors;
- user feedback.

The Streamlit monitoring dashboard will show:

- total runs;
- successful and failed runs;
- average latency;
- average evaluation score;
- feedback ratio;
- most-used datasets;
- weak retrieval cases;
- questions with poor groundedness.

---

## 17. Screenshots

The final README should include screenshots of:

1. the AOI selection map;
2. the generated answer and sources;
3. the evaluation page;
4. the monitoring dashboard;
5. the user feedback controls.

Suggested folder:

```text
screenshots/
```

Suggested filenames:

```text
01_aoi_interface.png
02_answer_with_sources.png
03_evaluation_dashboard.png
04_monitoring_dashboard.png
05_feedback_controls.png
```

---

## 18. Demo Video

A short application preview may be added to the README.

Suggested flow:

1. draw an AOI;
2. select crop and season;
3. ask a question;
4. show the answer and sources;
5. open the evaluation page;
6. open the monitoring page;
7. submit feedback.

A one-to-three-minute recording is sufficient.

---

## 19. Current Limitations

- The starter catalogue is small and illustrative.
- AOI selection currently narrows the user context but does not yet perform full spatial intersection against all dataset geometries.
- Document ingestion and vector retrieval are not yet included in the starter MVP.
- The generated answer depends on the quality of the curated catalogue and retrieved documents.
- Local LLMs may produce technically incorrect recommendations.
- LLM-as-a-judge scores are approximate and should not be treated as expert validation.
- No full satellite-scene download or processing is included.
- No remote-sensing foundation model is trained or fine-tuned.

---

## 20. Roadmap

### Phase 1 — Starter MVP

- [x] Streamlit interface
- [x] AOI map drawing
- [x] Crop, season, and application filters
- [x] Curated dataset catalogue
- [x] Local Ollama generation
- [x] DuckDB run logging
- [x] Recent-runs table

### Phase 2 — RAG

- [ ] Collect and document knowledge sources
- [ ] Build document ingestion pipeline
- [ ] Chunk documents and create embeddings
- [ ] Store vectors in Chroma or Qdrant
- [ ] Retrieve relevant chunks
- [ ] Generate answers with source references

### Phase 3 — Evaluation and feedback

- [ ] Create evaluation dataset
- [ ] Add retrieval metrics
- [ ] Add LLM-as-a-judge
- [ ] Add thumbs up/down feedback
- [ ] Store comments and feedback statistics

### Phase 4 — Monitoring and documentation

- [ ] Add monitoring dashboard
- [ ] Add screenshots
- [ ] Add setup and usage documents
- [ ] Add demo video
- [ ] Test installation from a clean environment

### Optional future extension

- [ ] LangGraph orchestration
- [ ] precomputed GeoAI model outputs
- [ ] remote-sensing image-patch analysis
- [ ] Neo4j GeoAI knowledge graph in Capstone 2

---

## 21. Capstone Scope Decisions

The first capstone intentionally excludes:

- training a remote-sensing foundation model;
- downloading and processing full satellite scenes;
- Airflow or Kestra;
- Neo4j;
- Kubernetes;
- complex multi-agent autonomy.

This keeps the project focused on delivering a complete, evaluated, monitored, and reproducible LLM application within the available time.

---

## 22. License and Data Use

Before publishing the final project:

- verify the licence and reuse conditions of every source document;
- keep links and metadata for all public documentation;
- avoid redistributing copyrighted material when reuse is restricted;
- publish only synthetic or openly licensed local datasets;
- clearly distinguish curated metadata from official source data.

A project licence will be added before final submission.

---

## 23. Author

**Hanaa Hammad**

LLM Zoomcamp Capstone Project  
GeoAI, remote sensing, RAG, local LLMs, evaluation, and monitoring
