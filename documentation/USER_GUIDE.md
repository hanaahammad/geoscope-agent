# GeoScope Agent — User Guide

> **GeoScope helps Earth Observation researchers understand how AI can support their workflows — without replacing domain expertise.**

---

## 1. Purpose of This Guide

GeoScope is designed to help Earth Observation researchers and analysts explore how AI can support their analytical workflow.

It is **not** intended to replace remote-sensing expertise or make autonomous scientific decisions.

GeoScope combines:

```text
Earth Observation knowledge
+ geographic context
+ live satellite catalogue information
+ AI-assisted retrieval and generation
+ deterministic raster processing
+ evaluation
+ monitoring
+ governance
+ persistent workflow state
+ fixed and agentic orchestration
```

This guide explains the application page by page and provides a complete reviewer/demo workflow.

---

## 2. What You Can Do with GeoScope

With GeoScope, an Earth Observation researcher can:

- prepare and index technical PDF/HTML knowledge;
- define an AOI by drawing on a map or searching a place name;
- query live Sentinel-2 STAC data by date and cloud cover;
- distinguish scene-item count from distinct acquisition dates;
- generate Red, NIR, and NDVI GeoTIFF outputs;
- ask grounded Earth Observation questions using RAG;
- compare vector search, query rewriting, reranking, and Rewrite + Rerank;
- inspect retrieved evidence, vector ranks, final ranks, distances, and reranking scores;
- provide immediate human feedback;
- summarize or translate an existing grounded answer without rerunning retrieval;
- run retrieval evaluation using Hit Rate and MRR;
- run structured LLM-as-a-judge generation evaluation;
- inspect verdicts such as PASS / NEEDS REVIEW / FAIL;
- monitor runs, prompts, context, frameworks, traces, latency, quality signals, and governance metadata;
- hide historical/incomplete monitoring rows while keeping them stored;
- run an automated end-to-end demonstration with visible progress;
- save, resume, complete, and archive persistent analysis projects;
- compare a fixed LangChain pipeline with a bounded LangGraph agentic workflow;
- execute a deterministic natural-language GeoTIFF analysis from Sentinel-2 NDVI;
- generate a classified GeoTIFF and mapped class statistics;
- run GeoScope locally, validate it through Docker, and test Ollama Cloud integration separately.

The application is deliberately designed to show **where AI adds value and where a simpler deterministic workflow is preferable**.

---

## 3. Before Starting

### 3.1 Required local models

GeoScope uses Ollama locally by default.

```cmd
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

| Task | Model |
|---|---|
| Answer generation | `qwen2.5:7b-instruct` |
| Query rewriting | `qwen2.5:7b-instruct` |
| Embeddings | `nomic-embed-text` |
| LLM-as-a-judge | `llama3.1:8b` |

Start Ollama if necessary:

```cmd
ollama serve
```

### 3.2 Start GeoScope

```cmd
python -m streamlit run GeoScope.py
```

Then open:

```text
http://localhost:8501
```

### 3.3 Streamlit configuration

Create:

```text
.streamlit/secrets.toml
```

Typical local configuration:

```toml
GEOSCOPE_PROVIDER = "ollama"
OLLAMA_GENERATION_MODEL = "qwen2.5:7b-instruct"
OLLAMA_JUDGE_MODEL = "llama3.1:8b"
```

> Never commit `.streamlit/secrets.toml`.

---

## 4. Recommended GeoScope Workflow

```text
1. Prepare knowledge
       ↓
2. Define AOI + inspect Sentinel-2 availability
       ↓
3. Ask GeoAI
       ↓
4. Give immediate human feedback
       ↓
5. Evaluate retrieval and generation
       ↓
6. Monitor quality, context, frameworks and governance
       ↓
7. Run the Automated Demo
       ↓
8. Save / resume the project
       ↓
9. Compare fixed vs agentic orchestration
       ↓
10. Ask natural-language questions about a GeoTIFF
```

The steps do not always have to be executed in this exact order. GeoScope is deliberately exploratory.

---

# 5. Page 1 — Data Preparation

## Objective

Prepare the technical knowledge base used by GeoScope RAG.

```text
PDF / HTML
→ extract text
→ clean
→ chunk
→ embed
→ Chroma vector store
```

The user can upload or confirm knowledge documents, process them, and build the vector index.

The LLM should not be expected to answer Earth Observation questions from model memory alone. GeoScope retrieves relevant technical evidence first and gives it to the LLM as context.

---

# 6. Page 2 — AOI and STAC

## Objective

Connect the AI assistant to the real geographic and satellite-data context.

### Define an AOI

An AOI can be defined by:

- drawing on the map;
- searching a place name such as `Kom Ombo, Aswan, Egypt`.

### Search Sentinel-2

Choose:

- start date;
- end date;
- maximum cloud cover.

GeoScope searches the live Earth Search STAC catalogue for Sentinel-2 Level-2A scenes intersecting the AOI.

### Important temporal rule

```text
10 STAC items
does not necessarily mean
10 different dates
```

Several items may be tiles from the same acquisition date.

GeoScope therefore calculates both scene-item count and distinct acquisition-date count before supporting a time-series recommendation.

### GeoTIFF processing

For one selected Sentinel-2 scene and the current AOI, GeoScope can generate:

- Red;
- NIR;
- NDVI.

```text
Selected STAC scene
      ↓
Select Red / NIR / NDVI
      ↓
Read source raster
      ↓
Clip to AOI
      ↓
Generate GeoTIFF
      ↓
Download / open in GIS
```

The current demonstration intentionally remains limited to one AOI + one selected scene + one product → one GeoTIFF.

GeoScope does not currently implement a full production chain such as multi-tile mosaicking, full cloud masking, or aligned multi-date raster cubes.

---

# 7. Page 3 — Ask GeoAI

## Objective

Ask a technical Earth Observation question and inspect how GeoScope builds a grounded answer.

### Retrieval approaches

| Option | What happens |
|---|---|
| Vector | Original query → semantic search |
| Rewrite | Query rewrite → semantic search |
| Rerank | Semantic candidates → FlashRank |
| Rewrite + Rerank | Query rewrite → semantic candidates → FlashRank |

For the strongest advanced retrieval path, use **Rewrite + Rerank**.

### Query rewriting

Query rewriting occurs **before retrieval**.

```text
Original:
How can I monitor drought?

Rewritten for retrieval:
Earth observation methods for drought monitoring using vegetation indices,
soil moisture, precipitation anomalies, Sentinel-2 and MODIS
```

The rewritten query is used only to search the knowledge base. It does not replace the user's original intent and it does not rewrite the final answer after generation.

### What the user can inspect

- original retrieval input;
- rewritten query;
- vector rank;
- final rank;
- vector distance;
- FlashRank rerank score;
- retrieved source text.

GeoScope exposes the observable processing pipeline and evidence selection. It does not claim to expose hidden LLM reasoning.

### Human feedback

The user can record:

- 👍 Yes
- 👎 No
- optional comment

Detailed automated evaluation remains on Page 4.

### Summarize and translate

The original grounded answer remains visible.

The user can additionally choose:

- Summarize;
- Translate;
- Summarize + Translate.

Supported UI languages include English, French, Arabic, Spanish, and German.

Retrieval is **not rerun** for this transformation.

---

# 8. Page 4 — Evaluation and Feedback

## Objective

Evaluate retrieval and answer generation systematically.

### Retrieval evaluation

Ground-truth questions:

```text
data/evaluation_questions.csv
```

Metrics:

- **Hit Rate** — whether the expected relevant document is retrieved;
- **Mean Reciprocal Rank (MRR)** — how high the expected result appears.

The same questions are evaluated through:

- vector;
- rewrite;
- rerank;
- rewrite + rerank.

### Questions requiring improvement

A question is flagged when the expected supporting document was not retrieved successfully. This indicates a retrieval issue; it is not automatically an answer-generation failure.

### LLM-as-a-judge

The judge evaluates:

- relevance;
- groundedness;
- completeness;
- technical correctness;
- citation quality;
- geographic relevance;
- overall assessment.

Judge model:

```text
llama3.1:8b
```

The evaluation is also summarized into an operational verdict:

```text
PASS
NEEDS REVIEW
FAIL
```

A diagnostic/failure category may also be recorded.

---

# 9. Page 5 — Monitoring and AI Governance

## Objective

Inspect what happened across GeoScope runs and understand operational quality, observability, and governance.

### Application vs framework vs execution mode

| Field | Meaning |
|---|---|
| Application | What GeoScope is doing |
| Framework | Which execution/orchestration layer is used |
| Execution mode | How that framework behaves for that run |

Examples:

| Run type | Application | Framework | Execution mode |
|---|---|---|---|
| Ask GeoAI | Crop monitoring / selected EO use case | Application RAG | Fixed RAG |
| Automated Demo | Automated GeoScope demo | Application RAG | Automated fixed RAG demo |
| Page 9 fixed | Framework comparison | LangChain | Fixed pipeline |
| Page 9 agentic | Framework comparison | LangGraph | Agentic bounded workflow |
| Page 10 | Vegetation condition classification | Application geospatial workflow | Deterministic raster classification |

This separation is intentional:

> Crop monitoring / Urban heat are EO use cases. LangChain / LangGraph are orchestration frameworks.

### Fully instrumented run metadata

A current run may contain:

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

Judge results and human feedback are associated through `run_id`.

### Historical / incomplete runs

Older records may predate the richer observability schema.

Page 5 provides:

```text
Show historical / incomplete runs (Not recorded)
```

Unchecked focuses on fully instrumented runs. Checked includes historical/incomplete rows. Historical data is not deleted.

### Governance dimensions

- **Groundedness** — answer supported by evidence.
- **Explainability** — visible query rewrite, ranks, reranking, sources, framework and trace.
- **Transparency** — model, prompt, retrieval, context, execution mode and limitations are visible.
- **Human oversight** — user feedback and validation.
- **Reliability** — retrieval metrics + LLM-as-a-judge.
- **Traceability** — run IDs, timestamps, traces, feedback and workflow events.
- **Responsible use** — safeguards against unsupported geospatial conclusions and over-reliance on AI.

---

# 10. Page 6 — Automated Demo

## Objective

Run a prepared end-to-end GeoScope scenario with minimal interaction and visible progress.

```text
Resolve AOI
→ STAC search
→ check distinct dates
→ rewrite query
→ retrieve
→ rerank
→ generate answer
→ optional GeoTIFF
→ monitoring log
```

Recommended place:

```text
Kom Ombo, Aswan, Egypt
```

The page shows visible progress with a step counter and progress bar so the workflow does not appear as a black box.

For a quick demonstration, GeoTIFF generation can remain disabled. Enable it for the fuller workflow when remote raster access is stable.

The run is logged as an **Application RAG** workflow, not as LangChain or LangGraph.

---

# 11. Page 7 — Projects and Workflows

## Objective

Turn GeoScope from a temporary session into a stateful analytical assistant.

Example:

```text
Project: Kom Ombo Wheat Monitoring

1. AOI defined
2. STAC search
3. Knowledge retrieval
4. GeoAI recommendation
5. GeoTIFF processing
6. Evaluation
7. Completion
```

A project can be created, saved, resumed, completed, and archived.

Possible persisted information includes:

- AOI;
- date range;
- STAC scenes;
- question;
- retrieval approach;
- retrieved sources;
- answer;
- GeoTIFF metadata;
- evaluation;
- project history.

GeoScope also distinguishes implemented EO processing from guided or partially implemented workflows so it does not claim every EO scenario is fully automated.

---

# 12. Page 8 — Cloud Deployment Test

## Objective

Experimentally validate whether GeoScope can use Ollama Cloud when the Streamlit application is deployed remotely.

Correct terminology:

```text
GeoScope application
→ deployed on Streamlit Community Cloud

Remote LLM inference
→ provided by Ollama Cloud
```

The page can test:

- API-key configuration;
- cloud connection;
- model discovery;
- generation;
- judge/query-rewrite suitability;
- embeddings;
- vector-index compatibility considerations.

This page is optional and isolated from the default local workflow.

---

# 13. Page 9 — Fixed Pipeline vs Agentic

## Objective

Demonstrate the difference between a predefined RAG pipeline and a bounded agentic workflow.

### Fixed LangChain pipeline

```text
Question
→ Rewrite
→ Retrieve
→ Rerank
→ Context
→ Generate
```

The order is predetermined and appropriate when the required steps are known and repeatable.

LangChain provides a structured, composable, reproducible way to connect the pipeline components. It does not make the model more accurate by itself.

### Agentic LangGraph workflow

```text
Question
→ Planner
→ Tool
→ Observation
→ Planner
→ next permitted action
→ Final answer
```

LangGraph provides stateful orchestration, conditional routing, bounded tool use, and visible execution traces.

The agent does not replace the underlying validated functions.

### Question types

**Simple knowledge question**

> What is NDVI and which Sentinel-2 bands are used?

Usually fixed pipeline.

**Knowledge synthesis**

> How can Sentinel-2 and Landsat be combined for long-term crop monitoring?

Usually fixed pipeline.

**Context-aware analysis**

> Check whether the current AOI has suitable Sentinel-2 scenes and tell me if NDVI analysis is possible.

Agentic orchestration can add value.

**Decision requiring context**

> What should I verify before recommending a time-series analysis for this AOI?

The agent may need to inspect AOI/STAC state before choosing a tool.

### Compare both

The clearest reviewer demonstration is to execute the **same question** through both paths.

Page 9 exposes latency, source count, answers, execution traces, and evidence.

### Monitoring integration

```text
LangChain:
application = Framework comparison
framework = LangChain
execution_mode = Fixed pipeline

LangGraph:
application = Framework comparison
framework = LangGraph
execution_mode = Agentic bounded workflow
```

This allows Page 5 to show LangChain and LangGraph as distinct execution frameworks.

> **Key takeaway:** use the simplest orchestration pattern that solves the task.

---

# 14. Page 10 — Ask Your GeoTIFF

## Objective

Query **actual GeoTIFF raster values in natural language**.

This page is different from Page 3:

```text
Page 3 — Ask GeoAI
→ questions over technical knowledge + EO context

Page 10 — Ask Your GeoTIFF
→ questions over actual raster values
```

The page can work with:

- an NDVI GeoTIFF generated from the current AOI and Sentinel-2 scene; or
- an uploaded `.tif` / `.tiff` file.

## User flow

```text
1. Choose raster source
       ↓
2. Inspect raster
       ↓
3. Ask a natural-language question
       ↓
4. GeoScope routes the question to a deterministic raster tool
       ↓
5. Raster code computes the result
       ↓
6. LLM explains only that computed result
```

The question field is always visible. It remains disabled until a raster is ready.

## Example questions

Supported examples include:

- `What is the average NDVI?`
- `What are the minimum and maximum values?`
- `What percentage of the AOI has NDVI above 0.6?`
- `What percentage of pixels are below 0.2?`
- `How many valid pixels are there?`
- `What is the CRS and pixel resolution?`
- `Summarize this raster.`

If the raster is explicitly treated as NDVI, the user can also ask questions such as:

- `How much of the raster has high vegetation signal?`
- `How much has low or moderate vegetation signal?`
- `Show the vegetation-condition class distribution.`

## Deterministic raster tools

GeoScope routes bounded NLQ requests to deterministic operations such as:

```text
raster metadata
raster mean / median / min / max
valid-pixel count
percentage above threshold
percentage below threshold
NDVI class distribution
raster summary
```

The numerical values come from raster processing code, **not from the LLM**.

## Role of the LLM

The LLM receives:

```text
user question
+ deterministic tool result
```

and produces a concise grounded explanation.

For example:

```text
Question:
What percentage of the AOI has NDVI above 0.6?

Deterministic raster tool:
percentage_above_threshold
→ 27.4%

LLM:
Approximately 27.4% of valid raster pixels have NDVI above 0.6.
```

The LLM is instructed not to invent pixel values, percentages, coordinates, or unsupported scientific conclusions.

## NDVI natural-language GeoTIFF analysis

The earlier natural-language GeoTIFF analysis remains available as **one raster-analysis capability**, not the main purpose of Page 10.

Current transparent NDVI classes are:

| NDVI range | Interpretation |
|---|---|
| `< 0.00` | Negative / non-vegetation signal |
| `0.00–0.20` | Very low vegetation signal |
| `0.20–0.40` | Low vegetation signal |
| `0.40–0.60` | Moderate vegetation signal |
| `≥ 0.60` | High vegetation signal |

These are demonstration thresholds. They are not universal agronomic rules and are not a crop-type or general land-cover model.

## Monitoring integration

Natural-language raster-query runs are logged as:

```text
Application = Natural-language GeoTIFF analysis
Framework = Application geospatial workflow
Execution mode = Natural-language raster query
```

The run can also record:

- model;
- prompt ID/version;
- raster tool used;
- raster filename;
- deterministic tool result;
- execution trace;
- runtime.

## Current NLQ scope

Supported now:

- band-1 statistics;
- raster metadata;
- threshold percentages;
- NDVI vegetation-signal classes;
- raster summaries.

Not yet supported:

- arbitrary zonal questions;
- directional questions such as `Where is vegetation weakest?`;
- robust comparison between two raster dates;
- free-form spatial reasoning over arbitrary regions inside the image.

GeoScope does not silently approximate unsupported spatial questions.


# 15. Docker Usage

For normal development:

```cmd
python -m streamlit run GeoScope.py
```

For reproducibility testing:

```cmd
docker compose up --build
```

Then open:

```text
http://localhost:8501
```

On Windows with Docker Desktop, the application container can connect to Ollama running on the host through:

```text
http://host.docker.internal:11434
```

---

# 16. Suggested 7-Minute Demonstration

### Minute 1 — Problem

Explain that GeoScope connects EO technical knowledge with real satellite availability, geographic context, and traceable AI engineering.

### Minute 2 — AOI / STAC

Define an AOI and search Sentinel-2.

Highlight:

```text
scene items ≠ distinct acquisition dates
```

### Minute 3 — Ask GeoAI

Run **Rewrite + Rerank** and show the rewritten query, ranks, reranking score, evidence, answer, and human feedback.

### Minute 4 — Evaluation / Monitoring

Show Hit Rate/MRR, LLM-as-a-judge, verdict, and Monitoring.

Explain:

```text
application ≠ framework ≠ execution mode
```

### Minute 5 — Automated Demo

Show the visible progress counter and guided end-to-end execution.

### Minute 6 — Pipeline vs Agentic

Run **Compare both** and show LangChain vs LangGraph traces and their Monitoring rows.

### Minute 7 — Ask Your GeoTIFF

Use a generated NDVI GeoTIFF or upload a raster.

Ask, for example:

```text
What percentage of the AOI has NDVI above 0.6?
```

Show:

```text
NLQ
→ deterministic raster tool
→ computed result
→ grounded LLM explanation
```

Then optionally show the NDVI vegetation-condition class distribution.

Emphasize that numerical raster analysis is deterministic and the LLM only explains the result.


---

# 17. Troubleshooting

## Ollama connection error

```cmd
ollama serve
```

Local endpoint:

```text
http://localhost:11434
```

Docker on Windows:

```text
http://host.docker.internal:11434
```

## FlashRank cannot download

Corporate SSL inspection can block first-time model downloads. GeoScope can use a local FlashRank cache. Do not commit the cache to Git.

## Rasterio / Docker native-library error

The Docker image must include the system libraries required by Rasterio/GDAL.

## Nominatim / remote raster SSL error

Corporate networks may use a self-signed root certificate. The correct production solution is to configure the trusted corporate CA.

## Monitoring shows `Not recorded`

The run probably predates the richer observability schema. Leave **Show historical / incomplete runs** unchecked to focus on current fully instrumented runs.

## LangChain / LangGraph do not appear in Monitoring

Run Page 9 using **Run fixed LangChain**, **Run agentic LangGraph**, or **Compare both**.

## Page 10 has no raster ready

Page 10 can use either:

- a GeoTIFF generated by GeoScope; or
- an uploaded `.tif` / `.tiff`.

If using the generated NDVI path, create/select the AOI first and generate the raster before asking a question.

## Page 10 cannot generate the NDVI raster

Check AOI, date range, cloud threshold, STAC access, and remote raster connectivity.

---

# 18. Current Limitations

- no multi-tile mosaicking;
- no full cloud-mask raster workflow;
- no aligned multi-date raster cube;
- no scheduled ingestion orchestration;
- raster access depends on network availability;
- query rewriting and reranking increase evaluation runtime;
- Ollama Cloud page is a deployment/integration test, not the default runtime;
- the LangGraph agent is deliberately bounded and is not a fully autonomous EO system;
- Page 10 NLQ is currently bounded to raster statistics, metadata, thresholds, NDVI class distribution, and summaries;
- NDVI vegetation-signal classes are not validated agronomic crop-health categories;
- context-token counts shown in Monitoring are estimates rather than exact tokenizer counts.

---

# 19. Final Perspective

GeoScope is intentionally more than a chatbot.

```text
knowledge retrieval
+ advanced retrieval
+ real geographic context
+ satellite catalogue search
+ deterministic raster processing
+ natural-language GeoTIFF analysis
+ human oversight
+ retrieval evaluation
+ LLM-as-a-judge
+ monitoring
+ prompt/context observability
+ AI governance
+ persistence
+ LangChain fixed orchestration
+ LangGraph bounded agentic orchestration
+ Docker reproducibility
```

The central design principle is:

> **AI should assist the researcher with evidence, context, and traceable tools — not hide the analytical process or replace domain expertise.**
