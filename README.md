GeoScope Agent

GeoScope Agent is an AI-assisted Earth Observation application designed to help researchers and analysts understand how Generative AI can support practical remote-sensing workflows.

The project combines technical-document retrieval, geographic context, live Sentinel-2 catalogue search, raster processing, evaluation, human feedback, monitoring, AI governance, persistent analysis workflows, and a comparison between fixed and agentic orchestration.

GeoScope was developed as a capstone project for the LLM Zoomcamp.

Note to Reviewer

GeoScope is implemented as a Streamlit application rather than a notebook so the complete AI-assisted Earth Observation workflow can be reviewed interactively.

Recommended review path:

AOI & STAC
→ Ask GeoAI
→ Evaluation
→ Monitoring / AI Governance
→ Projects & Workflows
→ Pipeline vs Agentic

GeoScope uses Ollama locally by default:

Role

Model

Generation

qwen2.5:7b-instruct

Query rewriting

qwen2.5:7b-instruct

Embeddings

nomic-embed-text

LLM-as-a-judge

llama3.1:8b

For a quick review, use a prepared AOI/demo scenario and select Rewrite + Rerank in Ask GeoAI. Inspect the rewritten query, ranking changes, retrieved evidence, and grounded answer; then review human feedback, LLM-as-a-judge, Monitoring / AI Governance, persistent Projects & Workflows, and finally the fixed-vs-agentic comparison.

Docker support is included for reproducibility. The Ollama Cloud page is an isolated deployment/integration test and is not required for the normal local workflow.

1. Problem and project objective

Earth Observation researchers often need to answer several connected questions:

Which sensor or dataset is appropriate for a task?

Which spectral bands or indices are relevant?

Are suitable scenes available for a selected location and period?

Do several returned scene items represent several dates, or only several tiles from one acquisition date?

Can a recommendation be translated into an executable raster product?

How can AI help without hiding the evidence or replacing domain expertise?

A conventional chatbot can answer a technical question without checking the real geographic context or the actual satellite catalogue.

GeoScope connects:

technical knowledge
+ AOI
+ live STAC catalogue
+ retrieval and reranking
+ LLM generation
+ geospatial processing
+ evaluation and governance

The objective is not to replace remote-sensing expertise. GeoScope demonstrates how AI can assist researchers with knowledge retrieval, imagery discovery, grounded recommendations, simple raster processing, evaluation, and traceable analytical workflows.

2. Why Streamlit instead of notebooks?

GeoScope is intentionally implemented as a Streamlit application rather than a collection of Jupyter notebooks.

A notebook is very useful for experimentation, but GeoScope is intended to demonstrate an end-to-end AI-assisted workflow that a researcher can use interactively without executing code cell by cell.

Notebook-style prototype
→ code-centric
→ cell-by-cell execution
→ suited to experimentation

GeoScope Streamlit application
→ user-centric
→ end-to-end workflow
→ persistent state
→ monitoring
→ governance
→ containerization
→ deployment-ready architecture

Using Streamlit also makes the application easier to demonstrate, evaluate, containerize, and eventually deploy.

3. What GeoScope does

GeoScope currently supports:

PDF and HTML document ingestion;

text extraction, cleaning, chunking, and embeddings;

persistent Chroma vector storage;

semantic vector retrieval;

LLM-based query rewriting;

FlashRank reranking;

comparison of four retrieval approaches;

local Ollama generation and judging;

optional OpenAI reviewer mode;

AOI drawing on an interactive map;

AOI search by place name;

live Sentinel-2 STAC search by AOI, date, and cloud cover;

distinct acquisition-date validation;

Red, NIR, and NDVI GeoTIFF generation;

retrieval evaluation with Hit Rate and MRR;

LLM-as-a-judge generation evaluation;

human thumbs-up / thumbs-down feedback and comments;

answer summarization and translation without rerunning retrieval;

DuckDB/dlt logging;

Streamlit monitoring and AI-governance metrics;

persistent projects that can be saved, resumed, completed, and archived;

a LangChain fixed-pipeline implementation;

a bounded LangGraph agentic workflow;

a side-by-side fixed-vs-agentic comparison;

Docker containerization;

an isolated Ollama Cloud deployment test page.

4. Understanding the Application

GeoScope is designed as an interactive Earth Observation workflow rather than a standalone chatbot. The application connects technical knowledge, geographic context, live Sentinel-2 catalogue search, AI-assisted retrieval, geospatial processing, evaluation, governance, and persistent analysis workflows.

### 4.1 Earth Observation context

The image below illustrates the kind of real-world Earth Observation context GeoScope is designed to support: agricultural areas, water bodies, desert environments, and spatially explicit analysis in Egypt.

![Earth Observation context in Egypt](documentation/images/earth_observation_context_egypt.jpeg)

### 4.2 Application workflow

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
Projects & Workflows
      ↓
Pipeline vs Agentic
```

### 4.3 Application pages

| Page | Purpose |
|---|---|
| **1 — Data Preparation** | Ingest PDF/HTML knowledge, clean and chunk text, create embeddings, and build the Chroma vector index. |
| **2 — AOI & STAC** | Define an Area of Interest, search live Sentinel-2 scenes, inspect distinct acquisition dates, and optionally generate GeoTIFF outputs. |
| **3 — Ask GeoAI** | Ask Earth Observation questions, select the retrieval strategy, inspect evidence and rankings, generate a grounded answer, provide feedback, and optionally summarize or translate the answer. |
| **4 — Evaluation & Feedback** | Compare retrieval approaches using Hit Rate and MRR and evaluate generated answers with an LLM-as-a-judge. |
| **5 — Monitoring** | Review run history, quality signals, feedback, traceability, and AI-governance indicators. |
| **6 — Automated Demo** | Run a prepared end-to-end GeoScope workflow for demonstration and peer review. |
| **7 — Projects & Workflows** | Save, resume, complete, and archive persistent GeoScope analysis projects. |
| **8 — Cloud Deployment Test** | Test Ollama Cloud as an optional remote inference provider for a remotely deployed Streamlit application. |
| **9 — Pipeline vs Agentic** | Compare a fixed LangChain RAG pipeline with a bounded LangGraph agentic workflow and understand when each is appropriate. |

The application is organized as a workflow rather than as a single chat screen. A researcher can move from knowledge preparation and geographic context to grounded AI assistance, evaluation, governance, and persistent project state.

5. Retrieval approaches

GeoScope implements four real retrieval pipelines:

Approach

Flow

Purpose

Vector

Original query → Chroma

Semantic baseline

Rewrite

Rewritten query → Chroma

Improve retrieval wording

Rerank

Original query → Chroma candidates → FlashRank

Improve candidate ordering

Rewrite + Rerank

Rewrite → Chroma candidates → FlashRank

Full advanced retrieval pipeline

The Ask GeoAI page exposes the retrieval process so the user can inspect:

original retrieval input;

rewritten query;

vector rank;

final rank;

vector distance;

FlashRank score;

source text.

This supports transparency and explainability: the user can inspect how evidence was selected before the LLM produced an answer.

6. Fixed pipeline vs Agentic AI

GeoScope deliberately demonstrates that not every task requires Agentic AI.

Fixed LangChain pipeline

Question
→ Rewrite
→ Retrieve
→ Rerank
→ Build context
→ Generate grounded answer

The application defines the sequence. It is predictable, reproducible, and easy to evaluate.

Agentic LangGraph workflow

Question
→ Planner
→ Choose bounded tool
→ Observe tool result
→ Planner
→ Choose next action
→ Final answer

The planner can select among bounded actions such as:

inspect current geographic context;

search STAC;

retrieve technical knowledge;

produce the final answer.

The underlying STAC, retrieval, reranking, and generation functions remain explicit tools. The agent decides which action is needed next rather than replacing the validated operations.

Example question types:

Question type

Recommended approach

“What is NDVI?”

Fixed pipeline

Technical knowledge synthesis

Usually fixed pipeline

Check current AOI and imagery availability before recommending an analysis

Agentic workflow can add value

Decide whether the current context supports time-series analysis

Agentic workflow can inspect state and choose tools

7. GeoTIFF processing

What is a GeoTIFF?

A GeoTIFF is a raster image that contains pixel values and geographic reference information, including the coordinate system and spatial extent.

In GeoScope, a selected Sentinel-2 scene can be clipped to the current AOI and exported as:

Red;

NIR;

NDVI.

STAC scene
   ↓
Select Red / NIR / NDVI
   ↓
Clip to AOI
   ↓
Generate GeoTIFF
   ↓
Open in GIS / download

The file can later be opened in software such as QGIS or ArcGIS and remains correctly positioned geographically.

Current scope:

one AOI + one selected STAC scene + one product
→ one clipped GeoTIFF

GeoScope does not yet implement a complete production remote-sensing chain such as multi-tile mosaicking, cloud masking, or aligned multi-date raster cubes.

8. AI Governance

GeoScope treats AI governance as part of the operational lifecycle rather than as a policy statement.

Governance dimension

Applicability

GeoScope implementation

Groundedness

High

RAG evidence + groundedness evaluation

Explainability

High

Visible query rewrite, ranks, reranking scores, sources

Transparency

High

Provider/model/retrieval strategy and limitations are visible

Human oversight

High

👍 / 👎 feedback and comments at the point of use

Reliability / quality

High

Retrieval metrics + LLM-as-a-judge

Traceability / auditability

High

DuckDB/dlt logs + persistent workflow history

Ethical / responsible use

Context-specific

Public Earth-observation data; safeguards focus on avoiding unsupported geospatial conclusions

GeoScope does not perform individual profiling or demographic decision-making. Therefore, demographic fairness metrics are not the primary concern for this use case.

Responsible-use controls focus on:

public Earth-observation and technical sources;

evidence-backed answers;

explicit geographic context;

human validation;

source inspection;

temporal consistency checks;

traceable runs and workflow state.

An important domain guardrail is:

Number of STAC scene items ≠ number of acquisition dates

Several items can be tiles from the same day. GeoScope counts distinct acquisition dates and does not treat multiple same-date tiles as a time series.

9. Evaluation and feedback

Retrieval evaluation

ground-truth questions: data/evaluation_questions.csv;

metrics: Hit Rate and Mean Reciprocal Rank (MRR);

same ground truth evaluated across the four retrieval approaches.

Generation evaluation

GeoScope uses an independent LLM-as-a-judge to score:

relevance;

groundedness;

completeness;

technical correctness;

citation quality;

geographic relevance;

overall result.

Human feedback

Human feedback is collected directly on Ask GeoAI:

👍 Yes;

👎 No;

optional comment.

The feedback and evaluation records are persisted through the dlt/DuckDB logging layer and surfaced in Monitoring.

10. Answer summarization and translation

After a grounded answer is generated, the user can optionally:

summarize it;

translate it;

summarize and translate it.

Supported UI choices currently include English, French, Arabic, Spanish, and German.

This is post-processing of the already grounded answer:

Grounded answer
→ optional summarize / translate

Retrieval is not rerun and the original long answer remains visible.

The transformation prompt is instructed to preserve Earth Observation terminology, dataset names, band names, indices, numerical values, limitations, and source references.

11. Persistent Projects and Workflows

GeoScope analyses can be stored as persistent workflow instances using DuckDB.

Example:

Project: Kom Ombo Wheat Monitoring 2026

1. AOI defined                  ✓
2. STAC search                  ✓
3. Knowledge retrieval          ✓
4. GeoAI recommendation         ✓
5. GeoTIFF processing           ▶
6. Evaluation                   ○
7. Completion                   ○

The project can be:

created;

saved;

resumed later;

completed;

archived.

Persisted state can include AOI, dates, STAC results, question, retrieval strategy, sources, answer, GeoTIFF metadata, evaluation, artifacts, and event history.

This makes GeoScope a stateful analytical assistant, not only a stateless chatbot.

12. Project structure

GeoScope_Agent/
├── GeoScope.py
├── pages/
│   ├── 1_Data_Preparation.py
│   ├── 2_AOI_and_STAC.py
│   ├── 3_Ask_GeoAI.py
│   ├── 4_Evaluation_and_Feedback.py
│   ├── 5_Monitoring.py
│   ├── 6_Automated_Demo.py
│   ├── 7_Projects_and_Workflows.py
│   ├── 8_Cloud_Deployment_Test.py
│   └── 9_Pipeline_vs_Agentic.py
├── src/
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
├── data/
│   ├── evaluation_questions.csv
│   └── demo/
├── documentation/
│   └── USER_GUIDE.md
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
└── README.md

Runtime files such as Chroma indexes, workflow DuckDB files, logs, secrets, and the FlashRank cache should not be committed.

13. Prerequisites

Recommended local setup:

Python 3.11;

Ollama installed and running;

internet access for Earth Search STAC and optional Nominatim place search;

FlashRank reranking model available locally or downloadable from the configured source.

14. Ollama local setup

GeoScope uses Ollama local by default for private and offline-friendly LLM inference.

Install Ollama, then pull the models:

ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
ollama pull llama3.1:8b

Model roles:

Role

Model

Main generation

qwen2.5:7b-instruct

Query rewriting

qwen2.5:7b-instruct

Embeddings

nomic-embed-text

LLM-as-a-judge

llama3.1:8b

The separation is intentional:

the generation model produces the user-facing answer;

the embedding model powers semantic search;

the judge model evaluates answers independently.

If Ollama is not already running:

ollama serve

Default local endpoint:

http://localhost:11434

15. Local installation

git clone <YOUR_REPOSITORY_URL>
cd GeoScope_Agent

py -3.11 -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Create:

.streamlit/secrets.toml

Example local configuration:

GEOSCOPE_PROVIDER = "ollama"
OLLAMA_GENERATION_MODEL = "qwen2.5:7b-instruct"
OLLAMA_JUDGE_MODEL = "llama3.1:8b"

Never commit secrets.toml.

Then run:

python -m streamlit run GeoScope.py

Open:

http://localhost:8501

16. FlashRank model

GeoScope uses:

ms-marco-MiniLM-L-12-v2

The local cache is intentionally excluded from Git.

Typical local structure:

data/
└── flashrank_cache/
    └── ms-marco-MiniLM-L-12-v2/
        ├── config.json
        ├── flashrank-MiniLM-L-12-v2_Q.onnx
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── tokenizer.json

Recommended .gitignore entry:

data/flashrank_cache/

17. Docker

GeoScope is containerized.

The application container does not need to contain the Ollama models. On Windows with Docker Desktop, GeoScope can call Ollama running on the host through:

http://host.docker.internal:11434

Typical startup:

docker compose up --build

Then open:

http://localhost:8501

For active development, running Streamlit directly is faster. Docker is mainly used to verify reproducibility and deployment packaging.

18. Ollama Cloud / Streamlit Community Cloud

Page 8 — Cloud Deployment Test is intentionally isolated from the current local runtime.

It demonstrates the deployment pattern:

GeoScope application
→ deployed on Streamlit Community Cloud

Remote inference
→ Ollama Cloud API

Correct terminology:

GeoScope is deployed on Streamlit Community Cloud and uses Ollama Cloud for remote model inference.

The page tests:

API authentication;

cloud model discovery;

generation;

query rewriting / judge suitability;

embeddings;

compatibility considerations with the existing Chroma index.

The cloud test is optional and does not replace the default local Ollama workflow.

19. Quick start for reviewers

Recommended demo path:

Start Ollama and the required models.

Run python -m streamlit run GeoScope.py.

Open Data Preparation and confirm/build the knowledge index.

Open AOI and STAC.

Draw an AOI or search for a place such as Kom Ombo, Aswan, Egypt.

Select a date range and cloud threshold.

Search Sentinel-2 scenes and inspect distinct dates.

Optionally generate Red, NIR, or NDVI GeoTIFF.

Open Ask GeoAI.

Select Rewrite + Rerank.

Ask a technical EO question.

Inspect query rewriting, ranks, reranking score, evidence, and answer.

Give human feedback.

Optionally summarize or translate the answer.

Open Evaluation and Feedback and run retrieval / LLM-as-a-judge evaluation.

Open Monitoring → AI Governance.

Open Projects and Workflows to save/resume an analysis.

Open Pipeline vs Agentic and compare both orchestration styles on the same question.

For a shorter guided flow, use Automated Demo.

20. Corporate-network SSL limitation

Some corporate networks inject a self-signed certificate into HTTPS connections. This can affect:

Nominatim geocoding;

remote Sentinel-2 GeoTIFF access;

first-time model downloads.

The preferred production solution is to configure the organization's trusted root CA.

Temporary insecure SSL workarounds should remain local and must not be presented as a production security configuration.

21. Current limitations

no multi-tile mosaicking;

no full cloud-mask raster workflow;

no aligned multi-date raster cube;

no scheduled ingestion orchestration;

automated raster access depends on network connectivity;

query rewriting and reranking increase evaluation runtime;

Ollama Cloud page is a deployment/integration test, not the default runtime;

the agent is deliberately bounded and is not intended as a fully autonomous Earth Observation system.

22. Rubric self-check

Criterion

GeoScope implementation

Dataset / source

Technical documents + live Earth Search STAC

Ingestion / API

PDF/HTML ingestion + STAC API

Application flow

Retrieval → context → prompt → LLM + geospatial tools

Retrieval evaluation

Hit Rate + MRR

LLM evaluation

LLM-as-a-judge

Human feedback

👍 / 👎 + comments

Interface

Streamlit multipage application

Monitoring

DuckDB/dlt + Streamlit dashboard

Advanced retrieval

Query rewriting + FlashRank reranking

Multiple retrieval approaches

Four pipelines

Geospatial processing

AOI + STAC + NDVI + GeoTIFF

Persistence

Projects & Workflows

AI governance

Explainability, transparency, oversight, traceability, quality

Agentic AI

Bounded LangGraph workflow

Framework orchestration

LangChain fixed pipeline

Containerization

Docker + Docker Compose

Reproducibility

README + user guide + requirements + configuration instructions

23. Security and Git hygiene

Keep these outside Git:

.venv/
__pycache__/
*.pyc
.streamlit/secrets.toml
.env
logs/*.duckdb
data/vector_store/
data/flashrank_cache/
data/geoscope_workflows.duckdb

24. Detailed usage

See the full guide:

documentation/USER_GUIDE.md