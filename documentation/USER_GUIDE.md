GeoScope Agent — User Guide

1. Purpose of this guide

GeoScope is designed to help Earth Observation researchers and analysts explore how AI can support their analytical workflow.

It is not intended to replace remote-sensing expertise or to make autonomous scientific decisions.

GeoScope combines:

Earth Observation knowledge
+ geographic context
+ live satellite catalogue information
+ AI-assisted retrieval and generation
+ simple raster processing
+ evaluation
+ governance
+ persistent workflow state

This guide explains the application page by page and suggests a complete demonstration workflow.

2. What you can do with GeoScope

With GeoScope, an Earth Observation researcher can:

prepare and index technical PDF/HTML knowledge;

define an AOI by drawing on a map or searching a place name;

query live Sentinel-2 STAC data by date and cloud cover;

distinguish scene-item count from distinct acquisition dates;

generate Red, NIR, and NDVI GeoTIFF outputs;

ask grounded Earth Observation questions using RAG;

compare vector search, query rewriting, reranking, and the full retrieval pipeline;

inspect retrieved evidence, vector ranks, final ranks, and reranking scores;

provide immediate human feedback;

summarize or translate an existing grounded answer without rerunning retrieval;

run retrieval evaluation using Hit Rate and MRR;

run structured LLM-as-a-judge generation evaluation;

monitor runs, quality signals, and AI-governance metrics;

save, resume, complete, and archive persistent analysis projects;

compare a fixed LangChain pipeline with a bounded LangGraph agentic workflow;

run GeoScope locally, validate it through Docker, and test Ollama Cloud integration separately.

The application is deliberately designed to show where AI adds value and where a simpler deterministic workflow is preferable.

3. Before starting

Required local models

GeoScope uses Ollama locally by default.

Install/pull:

ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
ollama pull llama3.1:8b

Roles:

Task

Model

Answer generation

qwen2.5:7b-instruct

Query rewriting

qwen2.5:7b-instruct

Embeddings

nomic-embed-text

LLM-as-a-judge

llama3.1:8b

Start Ollama if necessary:

ollama serve

Start GeoScope

python -m streamlit run GeoScope.py

Then open:

http://localhost:8501

3. Recommended GeoScope workflow

A complete research-assisted workflow looks like this:

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
6. Monitor quality and governance
       ↓
7. Save / resume the project
       ↓
8. Compare fixed vs agentic orchestration

The steps do not always have to be executed in this exact order. The application is deliberately exploratory.

4. Page 1 — Data Preparation

Objective

Prepare the technical knowledge base used by GeoScope RAG.

Typical flow:

PDF / HTML
→ extract text
→ clean
→ chunk
→ embed
→ Chroma vector store

The user can upload or confirm knowledge documents, process them, and build the vector index.

Why this matters

The LLM should not be expected to answer Earth Observation questions from model memory alone.

GeoScope retrieves relevant technical evidence first and gives it to the LLM as context.

5. Page 2 — AOI and STAC

Objective

Connect the AI assistant to the real geographic and satellite-data context.

Define an AOI

An AOI (Area of Interest) can be defined by:

drawing on the map;

searching a place name, for example Kom Ombo, Aswan, Egypt.

Search Sentinel-2

Choose:

start date;

end date;

maximum cloud cover.

GeoScope searches the live Earth Search STAC catalogue for Sentinel-2 Level-2A scenes intersecting the AOI.

Important temporal rule

Do not confuse scene count with temporal observations.

10 STAC items
does not necessarily mean
10 different dates

Several returned items may be tiles from the same acquisition date.

GeoScope therefore calculates:

number of scene items;

number of distinct acquisition dates.

A time-series recommendation should only be considered when there is sufficient temporal coverage.

GeoTIFF processing

What is a GeoTIFF?

A GeoTIFF is a raster file containing both:

pixel values;

geographic reference information.

This means the output knows where it belongs on the Earth.

What GeoScope can generate

For one selected Sentinel-2 scene and the current AOI, GeoScope can generate:

Red;

NIR;

NDVI.

Flow:

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

The result can be opened in QGIS or ArcGIS.

Current processing scope

The demonstration intentionally remains simple:

one AOI
+ one selected scene
+ one product
→ one GeoTIFF

It does not yet implement a full production processing chain such as cloud masking, mosaicking, or multi-date raster cubes.

6. Page 3 — Ask GeoAI

Objective

Ask a technical Earth Observation question and inspect how GeoScope builds a grounded answer.

The page deliberately shows more than the final answer.

Retrieval approaches

Option

What happens

Vector

Original query → semantic search

Rewrite

Query rewrite → semantic search

Rerank

Semantic candidates → FlashRank

Rewrite + Rerank

Query rewrite → semantic candidates → FlashRank

For the strongest retrieval path, use:

Rewrite + Rerank

How to interpret the retrieval results

Original retrieval input
The user's original question plus the relevant GeoScope context.

Rewritten query
An LLM-generated version optimized for semantic retrieval. It is used for retrieving evidence and does not replace the user's original intent.

Vector rank
The initial position returned by semantic vector search.

Final rank
The position after FlashRank reranking.

Rerank score
A relevance estimate used to reorder the candidate evidence.

Why expose these details?

This supports AI governance:

Explainability
→ how was the evidence selected?

Transparency
→ which retrieval method was used?

GeoScope does not claim to expose the hidden internal reasoning of the LLM. It exposes the observable processing pipeline and evidence selection.

Human feedback

Immediately after reviewing the answer, the user can record:

👍 Yes
or
👎 No

and optionally provide a comment.

This is intentionally captured at the point of use.

The detailed automated evaluation remains on Page 4.

Summarize and translate

The original grounded answer stays visible.

The user can additionally choose:

Summarize;

Translate;

Summarize + Translate.

Target languages include:

English;

French;

Arabic;

Spanish;

German.

Flow:

Original grounded answer
       ↓
Optional transformation
       ↓
Additional summary / translation

GeoScope does not rerun retrieval for this transformation.

The transformation is instructed to preserve:

Earth Observation terminology;

sensor/dataset names;

bands;

spectral indices;

numbers;

limitations;

references.

7. Page 4 — Evaluation and Feedback

Objective

Evaluate the retrieval pipeline and the generated answer systematically.

Human feedback is already captured in Ask GeoAI. This page focuses on formal evaluation.

Retrieval evaluation

GeoScope uses a ground-truth question set:

data/evaluation_questions.csv

Metrics:

Hit Rate

Measures whether the expected relevant document appears in the retrieved results.

Mean Reciprocal Rank (MRR)

Measures how high the expected result appears in the ranked list.

GeoScope evaluates the same questions through:

vector;

rewrite;

rerank;

rewrite + rerank.

This allows a direct comparison of retrieval strategies.

LLM-as-a-judge

GeoScope uses a separate judge model to assess answer quality.

Current evaluation dimensions include:

relevance;

groundedness;

completeness;

technical correctness;

citation quality;

geographic relevance;

overall assessment.

The local judge model is:

llama3.1:8b

The objective is to complement, not replace, human evaluation.

8. Page 5 — Monitoring and AI Governance

Objective

Inspect what has happened across GeoScope runs and evaluate the quality/governance of the system.

Monitoring uses DuckDB with the project dlt logging layer.

Governance dimensions

Groundedness

Is the answer supported by retrieved evidence?

Explainability

Can the user inspect how evidence was selected?

GeoScope exposes:

rewritten query;

retrieval strategy;

source ranks;

reranking scores;

source text.

Transparency

GeoScope exposes the operational setup and important limitations rather than presenting the AI as a black box.

Human oversight

The researcher remains responsible for accepting or rejecting the recommendation.

GeoScope records human feedback.

Reliability

Retrieval metrics and LLM-as-a-judge scores provide systematic quality signals.

Traceability / auditability

Runs, timestamps, feedback, evaluation, and persistent workflow events allow the analytical path to be reconstructed.

Ethical / responsible use

This dimension is context-specific.

GeoScope primarily uses public Earth Observation and technical data and does not perform personal profiling or demographic decision-making.

Therefore, governance focuses on relevant risks such as:

unsupported geospatial conclusions;

misleading temporal interpretation;

hallucinated technical claims;

lack of evidence;

over-reliance on an AI answer.

9. Page 6 — Automated Demo

Objective

Run a prepared end-to-end GeoScope scenario with minimal interaction.

Typical flow:

Resolve AOI
→ STAC search
→ check distinct dates
→ rewrite query
→ retrieve
→ rerank
→ generate answer
→ optional GeoTIFF
→ monitoring log

Recommended demo place:

Kom Ombo, Aswan, Egypt

For a quick demonstration, GeoTIFF generation can remain disabled.

For a full demonstration, enable raster generation after checking network access.

10. Page 7 — Projects and Workflows

Objective

Turn GeoScope from a temporary chat/session into a stateful analytical assistant.

An analysis can be represented as a persistent project:

Project: Kom Ombo Wheat Monitoring

1. AOI defined
2. STAC search
3. Knowledge retrieval
4. GeoAI recommendation
5. GeoTIFF processing
6. Evaluation
7. Completion

The project can be:

created;

saved;

resumed;

completed;

archived.

Why this matters

A researcher may not complete an analysis in one session.

Persistent workflows allow the user to return later and continue from the saved state rather than reconstructing the analysis manually.

Possible persisted information includes:

AOI;

date range;

STAC scenes;

question;

retrieval approach;

retrieved sources;

answer;

GeoTIFF metadata;

evaluation;

project history.

11. Page 8 — Cloud Deployment Test

Objective

Experimentally validate whether GeoScope could use Ollama Cloud when deployed on Streamlit Community Cloud.

Important terminology:

GeoScope application
→ deployed on Streamlit Community Cloud

LLM inference
→ provided by Ollama Cloud

GeoScope itself is not deployed on Ollama Cloud.

The page tests:

API-key configuration;

cloud connection;

model discovery;

generation;

judge/query-rewrite suitability;

embeddings.

This page is optional and isolated from the default local workflow.

12. Page 9 — Fixed Pipeline vs Agentic

Objective

Demonstrate the difference between a predefined RAG pipeline and a bounded agentic workflow.

Fixed LangChain pipeline

Question
→ Rewrite
→ Retrieve
→ Rerank
→ Context
→ Generate

The order is predetermined.

This approach is appropriate when the required steps are known.

What LangChain adds

LangChain provides a structured, composable, reproducible way to connect the pipeline components.

Using LangChain does not make the model more accurate by itself.

Agentic LangGraph workflow

Question
→ Planner
→ Tool
→ Observation
→ Planner
→ ...
→ Final answer

The planner chooses the next permitted action according to the question and current GeoScope state.

What LangGraph adds

LangGraph provides:

stateful orchestration;

conditional routing;

bounded tool use;

visible execution traces.

The agent does not replace the underlying validated functions.

Question types

The page includes prepared question categories.

Simple knowledge question

Example:

What is NDVI and which Sentinel-2 bands are used?

Usually a fixed pipeline is enough.

Knowledge synthesis

Example:

How can Sentinel-2 and Landsat be combined for long-term crop monitoring?

A fixed pipeline is usually sufficient.

Context-aware analysis

Example:

Check whether the current AOI has suitable Sentinel-2 scenes
and tell me if NDVI analysis is possible.

Agentic orchestration can add value because the assistant may need to inspect the current context.

Decision requiring context

Example:

What should I verify before recommending a time-series analysis
for this AOI?

The agent may need to inspect AOI/STAC state before deciding which tool to call.

Key takeaway

Use the simplest orchestration pattern that solves the task.

Agentic AI is useful when the next step depends on context. It is unnecessary when a predictable fixed pipeline already solves the problem.

13. Docker usage

GeoScope is containerized.

For normal development, local Streamlit is faster:

python -m streamlit run GeoScope.py

For final reproducibility testing:

docker compose up --build

Then open:

http://localhost:8501

On Windows with Docker Desktop, the GeoScope container can connect to Ollama running on the host through:

http://host.docker.internal:11434

The Ollama models do not need to be baked into the GeoScope image.

14. Suggested 6-minute demonstration

A concise peer-review demonstration could follow this sequence:

Minute 1 — Problem

Explain that GeoScope helps EO researchers connect technical knowledge with actual satellite availability and geographic context.

Minute 2 — AOI / STAC

Define an AOI and search Sentinel-2.

Highlight:

scene items ≠ distinct acquisition dates

Minute 3 — Ask GeoAI

Run Rewrite + Rerank.

Show:

rewritten query;

ranks;

reranking score;

sources;

grounded answer.

Record human feedback.

Minute 4 — Evaluation / Governance

Show LLM-as-a-judge and Monitoring → AI Governance.

Minute 5 — Persistence

Open Projects & Workflows.

Show that an analysis can be saved and resumed.

Minute 6 — Pipeline vs Agentic

Run the same context-aware question through:

Fixed LangChain;

Agentic LangGraph.

Explain why agentic AI is useful only when adaptive decisions are needed.

15. Troubleshooting

Ollama connection error

Check:

ollama serve

Then verify:

http://localhost:11434

For Docker on Windows:

http://host.docker.internal:11434

FlashRank cannot download

Corporate SSL inspection can block model downloads.

GeoScope supports using a local FlashRank cache.

The model cache should not be committed to Git.

Rasterio / Docker native-library error

The Docker image must include the system libraries required by Rasterio/GDAL.

If Docker works locally after the provided Dockerfile build, no application-code change is needed.

Nominatim / remote raster SSL error

A corporate network may use a self-signed root certificate.

The correct production solution is to configure the trusted corporate CA.

Temporary insecure SSL workarounds should not be considered production configuration.

16. Final perspective

GeoScope is intentionally more than a chatbot.

It demonstrates an AI-assisted Earth Observation workflow with:

knowledge retrieval
+ real geographic context
+ satellite catalogue search
+ raster processing
+ human oversight
+ automated evaluation
+ monitoring
+ AI governance
+ persistence
+ fixed and agentic orchestration

The central design principle is:

AI should assist the researcher with evidence, context, and traceable tools — not hide the analytical process or replace domain expertise.