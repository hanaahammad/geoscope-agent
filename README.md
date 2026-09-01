# GeoScope Agent

> **GeoScope helps Earth Observation researchers understand how AI can support their workflows — without replacing domain expertise.**

GeoScope Agent is an **AI-assisted Earth Observation application** developed as a capstone project for the **LLM Zoomcamp**.

It combines technical-document retrieval, geographic context, live Sentinel-2 catalogue search, raster processing, evaluation, human feedback, monitoring, AI governance, persistent analysis workflows, deterministic vegetation-condition classification, and a comparison between fixed and agentic orchestration.

![Earth Observation context in Egypt](documentation/images/earth_observation_context_egypt.jpeg)

---

## 👀 Note to Reviewer

GeoScope is implemented as a **Streamlit application rather than a notebook** so the complete workflow can be reviewed interactively.

### Recommended review path

**AOI & STAC** → **Ask GeoAI** → **Evaluation & Feedback** → **Monitoring** → **Automated Demo** → **Pipeline vs Agentic** → **Vegetation Condition Classification**

For a quick review:

1. Start Ollama and the required models.
2. Define or load an AOI.
3. Search Sentinel-2 scenes.
4. Open **Ask GeoAI** and select **Rewrite + Rerank**.
5. Inspect the rewritten query, ranking changes, retrieved evidence, and grounded answer.
6. Give human feedback.
7. Review **LLM-as-a-judge** and **Monitoring / AI Governance**.
8. Run the **Automated Demo** to see the end-to-end workflow with visible progress.
9. Open **Pipeline vs Agentic** and compare LangChain and LangGraph using the same question.
10. Open **Vegetation Condition Classification** to execute a deterministic EO task from AOI → NDVI → classified GeoTIFF.

> **Docker support is included for reproducibility.**  
> The **Ollama Cloud** page is an isolated deployment/integration test and is **not required** for the normal local workflow.

---

## 🎯 1. Project Objective

Earth Observation researchers often need to answer several connected questions:

- Which **sensor or dataset** is appropriate for a task?
- Which **spectral bands or indices** are relevant?
- Are suitable scenes available for a selected **location and period**?
- Do several STAC items represent **different dates**, or only several tiles from the same date?
- Can a recommendation be translated into an executable **raster product**?
- Can an assistant help execute a bounded EO task rather than only explain it?
- How can AI help without hiding the evidence or replacing domain expertise?

A conventional chatbot may answer a technical question without checking the real geographic context or the actual satellite catalogue.

GeoScope connects:

**Technical knowledge + AOI + live STAC catalogue + retrieval + LLM generation + geospatial processing + evaluation + monitoring + governance**

The objective is **not to replace remote-sensing expertise**. GeoScope demonstrates how AI can assist with knowledge retrieval, imagery discovery, grounded recommendations, simple raster processing, evaluation, traceable analytical workflows, and bounded EO task execution.

---

## 🖥️ 2. Why Streamlit Instead of Notebooks?

GeoScope is intentionally implemented as a **multipage Streamlit application** rather than a collection of Jupyter notebooks.

| Notebook-style prototype | GeoScope Streamlit application |
|---|---|
| Code-centric | User-centric |
| Cell-by-cell execution | End-to-end workflow |
| Excellent for experimentation | Better for demonstration and evaluation |
| Temporary notebook state | Persistent workflow state |
| Limited operational view | Monitoring and governance |
| Harder to package | Docker-ready |

This makes the project easier to **use, demonstrate, evaluate, containerize, and eventually deploy**.

---

## 🧭 3. Understanding the Application

GeoScope is designed as an **interactive workflow**, not as a single chatbot screen.

### 3.1 Main application flow

```text
Data Preparation
      ↓
AOI & STAC
      ↓
Ask GeoAI
      ↓
Evaluation & Feedback
      ↓
Monitoring & AI Governance
      ↓
Automated Demo
      ↓
Projects & Workflows
      ↓
Pipeline vs Agentic
      ↓
Vegetation Condition Classification
```

### 3.2 Application pages

| Page | What the user can do |
|---|---|
| **1 — Data Preparation** | Ingest PDF/HTML knowledge, clean and chunk text, create embeddings, and build the Chroma vector index. |
| **2 — AOI & STAC** | Define an AOI, search live Sentinel-2 scenes, inspect distinct acquisition dates, and generate GeoTIFF outputs. |
| **3 — Ask GeoAI** | Ask EO questions, choose a retrieval strategy, inspect evidence/ranks, generate a grounded answer, give feedback, summarize, or translate. |
| **4 — Evaluation & Feedback** | Compare retrieval strategies with Hit Rate/MRR and evaluate generated answers with an LLM-as-a-judge. |
| **5 — Monitoring** | Review instrumented runs, frameworks, prompt/context settings, latency, evaluation signals, feedback, traceability, and AI-governance indicators. |
| **6 — Automated Demo** | Run a prepared end-to-end scenario with visible step-by-step progress and monitoring logging. |
| **7 — Projects & Workflows** | Save, resume, complete, and archive persistent GeoScope projects. |
| **8 — Cloud Deployment Test** | Test Ollama Cloud as an optional remote inference provider. |
| **9 — Pipeline vs Agentic** | Compare a fixed LangChain pipeline with a bounded LangGraph agentic workflow and log both as fully instrumented runs. |
| **10 — Vegetation Condition Classification** | Use the current AOI and a Sentinel-2 scene to compute NDVI, classify vegetation signal, show statistics/map output, and download a classified GeoTIFF. |

---

## 🏗️ 4. Architecture

**GeoScope architecture**

```mermaid
graph TD
    A[PDF and HTML Knowledge] --> B[Extract Clean Chunk]
    B --> C[Embeddings]
    C --> D[Chroma Vector Store]

    Q[User Question] --> R[Query Rewriting]
    R --> D
    D --> V[Vector Candidates]
    V --> F[FlashRank Reranking]
    F --> G[Top Evidence]

    H[Area of Interest] --> I[Sentinel 2 STAC Search]
    I --> J[Distinct Date Check]

    G --> K[LLM Generation]
    J --> K
    K --> L[Grounded GeoAI Answer]

    L --> M[Human Feedback]
    L --> N[LLM Judge]

    M --> O[Monitoring Store]
    N --> O

    I --> P[Red NIR NDVI]
    P --> S[GeoTIFF]
    P --> T[Vegetation Classification]

    Q --> U[LangChain Fixed Pipeline]
    Q --> W[LangGraph Agent]
    U --> O
    W --> O
```

### Main technology components

- **Streamlit** — user interface
- **Chroma** — vector store
- **Ollama** — local LLM inference and embeddings
- **FlashRank** — reranking
- **Earth Search STAC** — live Sentinel-2 catalogue
- **Rasterio** — GeoTIFF processing
- **DuckDB + dlt** — logging, monitoring, evaluation history
- **LangChain** — fixed pipeline orchestration
- **LangGraph** — bounded agentic orchestration

---

## 🔎 5. Retrieval Approaches

GeoScope implements **four real retrieval pipelines**.

| Approach | Flow | Purpose |
|---|---|---|
| **Vector** | Original query → Chroma | Semantic baseline |
| **Rewrite** | Rewritten query → Chroma | Improve retrieval wording |
| **Rerank** | Original query → Chroma candidates → FlashRank | Improve candidate ordering |
| **Rewrite + Rerank** | Rewrite → Chroma candidates → FlashRank | Full advanced retrieval pipeline |

### What the user can inspect

The **Ask GeoAI** page exposes:

- **Original retrieval input**
- **Rewritten query**
- **Vector rank**
- **Final rank**
- **Vector distance**
- **FlashRank score**
- **Retrieved source text**

This supports **transparency and explainability**: the user can inspect how evidence was selected before the LLM generates the answer.

### Query rewriting happens before retrieval

GeoScope may reformulate a conversational or vague question into terminology that better matches the indexed technical documents.

Example:

```text
Original:
How can I monitor drought?

Rewritten for retrieval:
Earth observation methods for drought monitoring using vegetation indices,
soil moisture, precipitation anomalies, Sentinel-2 and MODIS
```

The rewritten query is used **only to search the knowledge base**. GeoScope does **not** rewrite the final answer after generation.

---

## 🌍 6. Earth Observation Integration

### 6.1 AOI and Sentinel-2 STAC search

The user can:

- draw an **Area of Interest (AOI)** on the map;
- search by place name;
- select a date range;
- filter by cloud cover;
- query live Sentinel-2 Level-2A scenes.

### 6.2 Important temporal guardrail

> **STAC scene items ≠ distinct acquisition dates**

Several scene items may represent different tiles from the **same day**.

GeoScope therefore counts **distinct acquisition dates** before recommending time-series analysis.

### 6.3 GeoTIFF processing

A **GeoTIFF** is a raster image that stores both pixel values and geographic reference information.

GeoScope can export a selected Sentinel-2 scene as:

- **Red**
- **NIR**
- **NDVI**

Workflow:

**Selected STAC scene** → **Choose Red/NIR/NDVI** → **Clip to AOI** → **Generate GeoTIFF** → **Open in QGIS/ArcGIS**

**Current scope:** one AOI + one selected scene + one product → one clipped GeoTIFF.

### 6.4 Vegetation-condition classification

Page 10 extends the NDVI workflow into a transparent, deterministic classification:

```text
AOI
→ Sentinel-2 scene
→ Red + NIR
→ NDVI
→ NDVI threshold classes
→ class statistics
→ interactive map
→ classified GeoTIFF
```

Current classes:

| NDVI range | Vegetation-signal class |
|---|---|
| `< 0.00` | Negative / non-vegetation signal |
| `0.00–0.20` | Very low vegetation signal |
| `0.20–0.40` | Low vegetation signal |
| `0.40–0.60` | Moderate vegetation signal |
| `≥ 0.60` | High vegetation signal |

This is deliberately described as **vegetation-signal classification**, not crop-type classification, validated crop-health assessment, or general land-cover mapping.

The LLM does **not** classify pixels. The raster calculation is performed deterministically by the geospatial code.

> GeoScope does not yet implement a complete production remote-sensing chain such as multi-tile mosaicking, full cloud masking, aligned multi-date raster cubes, or a validated supervised land-cover model.

---

## 🤖 7. Models and Ollama Setup

### 7.1 Models Used

GeoScope uses **Ollama locally by default**.

| Role | Model |
|---|---|
| **Generation** | `qwen2.5:7b-instruct` |
| **Query rewriting** | `qwen2.5:7b-instruct` |
| **Embeddings** | `nomic-embed-text` |
| **LLM-as-a-judge** | `llama3.1:8b` |

The separation is intentional:

- the **generation model** produces user-facing answers;
- the **embedding model** powers semantic retrieval;
- the **judge model** evaluates generated answers independently.

### 7.2 Ollama Local Setup

Install Ollama, then pull the models used by GeoScope:

```cmd
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

Check that they are available:

```cmd
ollama list
```

If Ollama is not already running:

```cmd
ollama serve
```

GeoScope expects Ollama locally at:

```text
http://localhost:11434
```

Create the Streamlit secrets file:

```text
.streamlit/secrets.toml
```

Example configuration:

```toml
GEOSCOPE_PROVIDER = "ollama"
OLLAMA_GENERATION_MODEL = "qwen2.5:7b-instruct"
OLLAMA_JUDGE_MODEL = "llama3.1:8b"
```

The embedding model used by the retrieval pipeline is:

```text
nomic-embed-text
```

Then start GeoScope:

```cmd
python -m streamlit run GeoScope.py
```

> **Note:** Never commit `.streamlit/secrets.toml`.

### 7.3 FlashRank Reranker

GeoScope uses:

```text
ms-marco-MiniLM-L-12-v2
```

The FlashRank cache is intentionally excluded from Git.

When a reviewer selects a reranking approach:

- GeoScope checks whether the local FlashRank cache is available;
- if it is missing, GeoScope can try FlashRank's normal first-time model initialization/download;
- if initialization is not possible, the application **does not crash with an unhandled traceback**;
- instead, it displays a clear message and asks the reviewer to use **Vector search** or **Query rewriting + vector search**, or to install/cache the reranker and retry.

This keeps the behavior transparent: GeoScope never silently claims that reranking was executed when the reranker was unavailable.

---

## 📊 8. Evaluation and Human Feedback

### 8.1 Retrieval evaluation

Ground-truth questions:

`data/evaluation_questions.csv`

Metrics:

- **Hit Rate**
- **Mean Reciprocal Rank (MRR)**

The same ground truth is evaluated across all four retrieval approaches.

### 8.2 LLM-as-a-judge

Generation evaluation includes:

- **Relevance**
- **Groundedness**
- **Completeness**
- **Technical correctness**
- **Citation quality**
- **Geographic relevance**
- **Overall assessment**

The evaluation page also converts judge scores into an operational verdict such as:

- **PASS**
- **NEEDS REVIEW**
- **FAIL**

with a diagnostic/failure category where applicable.

### 8.3 Human feedback

Human feedback is collected directly in **Ask GeoAI**:

- 👍 **Yes**
- 👎 **No**
- optional written comment

Feedback and evaluation records are persisted and surfaced in Monitoring.

---

## 🛡️ 9. Monitoring and AI Governance

GeoScope treats governance as part of the **operational workflow**, not as a separate policy document.

### 9.1 How GeoScope records an execution

Monitoring separates three concepts that should not be mixed:

| Field | Meaning |
|---|---|
| **Application** | What GeoScope is doing — for example Crop monitoring or Vegetation condition classification |
| **Framework** | Which execution/orchestration layer is used — for example Application RAG, LangChain, LangGraph |
| **Execution mode** | How that framework behaves for the run — for example Fixed RAG, Fixed pipeline, Agentic bounded workflow, Deterministic raster classification |

Examples:

| Run type | Application | Framework | Execution mode |
|---|---|---|---|
| Ask GeoAI | Crop monitoring / selected EO use case | Application RAG | Fixed RAG |
| Automated Demo | Automated GeoScope demo | Application RAG | Automated fixed RAG demo |
| Page 9 fixed | Framework comparison | LangChain | Fixed pipeline |
| Page 9 agentic | Framework comparison | LangGraph | Agentic bounded workflow |
| Page 10 | Vegetation condition classification | Application geospatial workflow | Deterministic raster classification |

This separation is intentional: **Crop monitoring and Urban heat are EO use cases. LangChain and LangGraph are orchestration frameworks.**

### 9.2 Instrumented run metadata

New instrumented runs can include:

```text
run_id
created_at

question
application

framework
execution_mode
model

prompt_id
prompt_version

retrieval_approach
original_query
rewritten_query
top_k
candidate_k

chunk_count
context_characters
estimated_context_tokens

answer
latency_seconds
status

trace
tool_calls
step_count
```

Judge results and human feedback are associated through the same `run_id`.

### 9.3 Historical / incomplete runs

Older runs were created before the richer observability fields existed.

Page 5 therefore provides:

```text
Show historical / incomplete runs (Not recorded)
```

When the checkbox is disabled, the dashboard focuses on fully instrumented runs. Historical data is **not deleted**.

### 9.4 Governance dimensions

| Governance dimension | How GeoScope addresses it |
|---|---|
| **Groundedness** | RAG evidence + groundedness evaluation |
| **Explainability** | Visible query rewrite, ranks, reranking scores, sources, framework and trace |
| **Transparency** | Provider, model, prompt version, retrieval strategy, context, framework and limitations are visible |
| **Human oversight** | Feedback and validation at the point of use |
| **Reliability / quality** | Retrieval metrics + LLM-as-a-judge |
| **Traceability / auditability** | DuckDB/dlt logs + run IDs + persistent workflow history |
| **Responsible use** | Evidence-backed answers, geographic context, temporal checks, deterministic EO processing, and human validation |

GeoScope primarily uses **public Earth Observation and technical data** and does not perform individual profiling or demographic decision-making.

The main risks are instead:

- unsupported geospatial conclusions;
- misleading temporal interpretation;
- overconfidence;
- hallucinated technical claims;
- missing evidence;
- confusing an LLM recommendation with a validated scientific result.

---

## ✍️ 10. Summarize and Translate

After a grounded answer is generated, the user can optionally:

- **Summarize**
- **Translate**
- **Summarize + Translate**

Supported UI choices currently include:

**English · French · Arabic · Spanish · German**

This is **post-processing of the existing grounded answer**. Retrieval is **not rerun**, and the original answer remains visible.

---

## 📂 11. Persistent Projects and Workflows

GeoScope can persist analytical workflow state in DuckDB.

Example:

| Step | Example status |
|---|---|
| AOI defined | ✅ Completed |
| STAC search | ✅ Completed |
| Knowledge retrieval | ✅ Completed |
| GeoAI recommendation | ✅ Completed |
| GeoTIFF processing | ▶ In progress |
| Evaluation | ○ Pending |
| Completion | ○ Pending |

Projects can be:

- **Created**
- **Saved**
- **Resumed**
- **Completed**
- **Archived**

Persisted state can include AOI, dates, STAC results, questions, retrieval strategy, sources, answers, GeoTIFF metadata, evaluation, artifacts, and event history.

> This makes GeoScope a **stateful analytical assistant**, not only a stateless chatbot.

---

## 🔀 12. Fixed Pipeline vs Agentic AI

GeoScope deliberately demonstrates that **not every AI workflow needs an agent**.

### 12.1 Fixed LangChain pipeline

```text
Question
→ Rewrite
→ Retrieve
→ Rerank
→ Build context
→ Generate
```

Use this when the required steps are **known and repeatable**.

The application defines the sequence, which makes the workflow predictable, reproducible, and easier to evaluate.

### 12.2 Agentic LangGraph workflow

```text
Question
→ Planner
→ Choose bounded tool
→ Observe
→ Planner
→ Next action
→ Answer
```

The planner can choose among bounded actions such as:

- inspect current geographic context;
- search STAC;
- retrieve technical knowledge;
- generate the final answer.

The underlying STAC, retrieval, reranking, and generation functions remain explicit tools. The planner decides **which permitted action is needed next** rather than replacing the validated operations.

### 12.3 Monitoring comparison

Page 9 writes fully instrumented runs to Monitoring:

```text
LangChain
framework = LangChain
execution_mode = Fixed pipeline

LangGraph
framework = LangGraph
execution_mode = Agentic bounded workflow
```

This allows Page 5 to compare framework usage, latency, retrieval/context metadata, and execution traces without mixing framework names into EO use-case labels.

> **Key takeaway:** Not every use case requires Agentic AI.  
> Use the simplest orchestration pattern that solves the task.

---

## ▶️ 13. Automated Demo

Page 6 provides a reviewer-friendly, one-button execution of the main workflow:

```text
AOI / place
→ Sentinel-2 STAC search
→ distinct-date validation
→ query rewriting
→ Chroma retrieval
→ FlashRank reranking
→ grounded answer
→ optional GeoTIFF
→ monitoring log
```

The page shows visible progress so the workflow is not a black box.

Example:

```text
Step 1/7 — Resolve Area of Interest
Step 2/7 — Search satellite catalogue
...
Step 7/7 — Complete
```

The demo is recorded as an **Application RAG** execution, not as LangChain or LangGraph. Page 9 contains the explicit framework comparison.

---

## 🌿 14. Vegetation Condition Classification

Page 10 demonstrates a bounded EO execution task:

> “Using the current AOI, find a suitable Sentinel-2 image and create a vegetation-condition classification from NDVI.”

The actual scientific processing is deterministic:

```text
AOI
→ Sentinel-2 scene
→ Red + NIR
→ NDVI
→ threshold classification
→ statistics
→ map
→ classified GeoTIFF
```

The page deliberately logs:

```text
Application = Vegetation condition classification
Framework = Application geospatial workflow
Execution mode = Deterministic raster classification
Model = No LLM used for pixel classification
```

This keeps the boundary clear between **AI orchestration/assistance** and **scientific raster processing**.

---

## 🗂️ 15. Project Structure

Rather than displaying the whole repository as one long line, the main structure is grouped by purpose:

```text
GeoScope_Agent/
│
├── GeoScope.py                         # Streamlit entry point
│
├── pages/                              # Application pages
│   ├── 1_Data_Preparation.py
│   ├── 2_AOI_and_STAC.py
│   ├── 3_Ask_GeoAI.py
│   ├── 4_Evaluation_and_Feedback.py
│   ├── 5_Monitoring.py
│   ├── 6_Automated_Demo.py
│   ├── 7_Projects_and_Workflows.py
│   ├── 8_Cloud_Deployment_Test.py
│   ├── 9_Pipeline_vs_Agentic.py
│   └── 10_Vegetation_Condition_Classification.py
│
├── src/                                # Core application logic
│   ├── ingest_documents.py
│   ├── build_vector_index.py
│   ├── retrieval.py
│   ├── query_rewrite.py
│   ├── reranking.py
│   ├── generation.py
│   ├── llm_provider.py
│   ├── geocoding.py
│   ├── stac_search.py
│   ├── geotiff_processing.py
│   ├── evaluation.py
│   ├── monitoring.py
│   ├── dlt_logging.py
│   ├── demo_runner.py
│   ├── workflow_store.py
│   ├── langchain_pipeline.py
│   ├── agentic_geoscope.py
│   └── ui.py
│
├── data/
│   ├── evaluation_questions.csv
│   └── demo/
│
├── documentation/
│   ├── USER_GUIDE.md
│   └── images/
│       └── earth_observation_context_egypt.jpeg
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
└── README.md
```

Runtime files such as Chroma indexes, workflow DuckDB files, logs, secrets, and the FlashRank model cache should **not** be committed.

---

## ⚙️ 16. Local Installation

### 16.1 Prerequisites

- **Python 3.11**
- **Ollama**
- Internet access for Earth Search STAC and optional Nominatim place search
- FlashRank model available locally or downloadable

### 16.2 Clone and create the environment

```cmd
git clone <YOUR_REPOSITORY_URL>
cd GeoScope_Agent

py -3.11 -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 16.3 Install Ollama models

```cmd
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

Start Ollama if necessary:

```cmd
ollama serve
```

Default endpoint:

```text
http://localhost:11434
```

### 16.4 Streamlit configuration

Create:

```text
.streamlit/secrets.toml
```

Example:

```toml
GEOSCOPE_PROVIDER = "ollama"
OLLAMA_GENERATION_MODEL = "qwen2.5:7b-instruct"
OLLAMA_JUDGE_MODEL = "llama3.1:8b"
```

> **Never commit `secrets.toml`.**

### 16.5 Run GeoScope

```cmd
python -m streamlit run GeoScope.py
```

Open:

`http://localhost:8501`

---

## 🐳 17. Docker

GeoScope is containerized.

The application container does **not** need to contain the Ollama models.

On Windows with Docker Desktop, the container can call Ollama running on the host through:

`http://host.docker.internal:11434`

Start:

```cmd
docker compose up --build
```

Open:

`http://localhost:8501`

For active development, running Streamlit directly is faster. Docker is mainly used to validate **reproducibility and packaging**.

---

## ☁️ 18. Ollama Cloud / Streamlit Community Cloud

Page **8 — Cloud Deployment Test** is intentionally isolated from the normal local runtime.

The page demonstrates/tests the deployment pattern:

```text
GeoScope application
→ Streamlit Community Cloud

Remote model inference
→ Ollama Cloud
```

The page can test:

- API authentication;
- cloud model discovery;
- generation;
- query-rewrite / judge suitability;
- embeddings;
- vector-index compatibility considerations.

This is an **optional deployment/integration test**, not the default GeoScope runtime.

---

## 🚀 19. Quick Start for Reviewers

1. Start Ollama and the required models.
2. Run `python -m streamlit run GeoScope.py`.
3. Open **Data Preparation** and confirm/build the knowledge index.
4. Open **AOI & STAC**.
5. Draw an AOI or search `Kom Ombo, Aswan, Egypt`.
6. Select a historical date range and cloud threshold.
7. Search Sentinel-2 and inspect **distinct acquisition dates**.
8. Optionally generate **Red, NIR, or NDVI GeoTIFF**.
9. Open **Ask GeoAI**.
10. Select **Rewrite + Rerank**.
11. Ask a technical EO question.
12. Inspect the rewritten query, ranks, reranking score, evidence, and answer.
13. Give human feedback.
14. Optionally summarize or translate.
15. Open **Evaluation & Feedback** and run the LLM-as-a-judge.
16. Open **Monitoring** and inspect the fully instrumented run.
17. Open **Automated Demo** for the guided end-to-end walkthrough.
18. Open **Pipeline vs Agentic** and select **Compare both**.
19. Return to **Monitoring** and verify that **LangChain** and **LangGraph** now appear as separate execution frameworks.
20. Open **Vegetation Condition Classification** and run the NDVI threshold-classification task.

---

## ⚠️ 20. Current Limitations

- No multi-tile mosaicking
- No full cloud-mask raster workflow
- No aligned multi-date raster cube
- No scheduled ingestion orchestration
- Automated raster access depends on network connectivity
- Query rewriting and reranking increase evaluation runtime
- Ollama Cloud page is a deployment/integration test, not the default runtime
- The LangGraph agent is deliberately bounded and is not a fully autonomous EO system
- Page 10 uses transparent NDVI thresholds rather than a trained crop-type or land-cover model
- NDVI vegetation-signal classes must not be presented as validated crop health without domain calibration
- Exact model-token counting is not performed; monitoring shows **estimated context tokens** based on context size

---

## 🔐 21. Security and Git Hygiene

Keep the following outside Git:

```gitignore
.venv/
venv/
env/
__pycache__/
*.pyc
.streamlit/secrets.toml
.env
logs/*.duckdb
data/vector_store/
data/flashrank_cache/
data/geoscope_workflows.duckdb
.vscode/
.idea/
```

---

## ✅ 22. Rubric Coverage

| Criterion | GeoScope implementation |
|---|---|
| **Dataset / source** | Technical documents + live Earth Search STAC |
| **Ingestion / API** | PDF/HTML ingestion + STAC API |
| **Application flow** | Retrieval → context → prompt → LLM + geospatial tools |
| **Vector database** | Chroma |
| **Retrieval evaluation** | Hit Rate + MRR |
| **LLM evaluation** | LLM-as-a-judge |
| **Human feedback** | 👍 / 👎 + comments |
| **Interface** | Streamlit multipage application |
| **Monitoring** | DuckDB/dlt + Streamlit Run Explorer |
| **Observability** | Framework, execution mode, prompt version, retrieval, context size, trace, latency |
| **Advanced retrieval** | Query rewriting + FlashRank reranking |
| **Multiple retrieval approaches** | Four real pipelines |
| **Geospatial processing** | AOI + STAC + NDVI + GeoTIFF |
| **Bounded EO task execution** | NDVI vegetation-condition classification + classified GeoTIFF |
| **Persistence** | Projects & Workflows |
| **AI governance** | Explainability, transparency, oversight, traceability, quality |
| **Agentic AI** | Bounded LangGraph workflow |
| **Framework orchestration** | LangChain fixed pipeline |
| **Framework comparison** | Same-question LangChain vs LangGraph execution + monitoring |
| **Containerization** | Docker + Docker Compose |
| **Reproducibility** | README + User Guide + requirements + configuration |

---

## 📖 23. Detailed User Guide

For the full page-by-page instructions, see:

**[documentation/USER_GUIDE.md](documentation/USER_GUIDE.md)**
