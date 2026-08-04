# GeoScope Agent

**GeoScope Agent** is an AI-assisted remote-sensing application that
combines technical-document retrieval, geographic context, live
Sentinel-2 catalogue search, raster processing, evaluation, feedback,
and monitoring.

It is a capstone project developed for the **LLM Zoomcamp**.

## Problem

Remote-sensing users often need to answer several connected questions:

- Which dataset or sensor should be used?
- Which spectral bands are relevant?
- Are suitable scenes available for the selected location and period?
- Do the returned scene items represent several dates or only several
  tiles from one date?
- Can the recommendation be translated into an executable raster
  product?

A conventional chatbot may answer the technical question without
checking actual data availability. GeoScope connects document knowledge
with an Area of Interest (AOI), a STAC catalogue, and a small executable
GeoAI workflow.

## What GeoScope does

GeoScope supports:

- PDF and HTML document ingestion;
- text extraction, cleaning, chunking, and embeddings;
- persistent Chroma vector storage;
- semantic vector retrieval;
- LLM-based query rewriting;
- FlashRank reranking;
- comparison of four retrieval approaches;
- local Ollama generation and judging;
- optional OpenAI reviewer mode;
- AOI drawing on an interactive map;
- AOI search by place name;
- Sentinel-2 STAC search by AOI, date, and cloud cover;
- distinct acquisition-date validation;
- Red, NIR, and NDVI GeoTIFF generation;
- retrieval evaluation with Hit Rate and MRR;
- LLM-as-a-judge generation evaluation;
- explicit thumbs-up/thumbs-down feedback;
- DuckDB/dlt run logging and a Streamlit monitoring dashboard;
- an end-to-end automated demonstration page.

## Architecture

```text
PDF / HTML documents
        │
        ▼
┌────────────────────────────┐
│ Document ingestion         │
│ extract → clean → chunk    │
│ Ollama embeddings          │
└─────────────┬──────────────┘
              ▼
       Chroma vector store
              │
User question │
      │       │
      ▼       │
Query rewriting
      │
      ▼
Vector candidate retrieval
      │
      ▼
FlashRank reranking
      │
      ▼
Top context chunks ──────────────────────┐
                                         │
AOI drawn or searched by text            │
      │                                  │
      ▼                                  │
Sentinel-2 STAC search                   │
date + cloud filters                     │
      │                                  │
      ▼                                  │
Scene count + distinct-date validation ──┤
                                         ▼
                                Ollama / OpenAI LLM
                                         │
                                         ▼
                              Grounded GeoAI answer
                                         │
                       ┌─────────────────┴──────────────┐
                       ▼                                ▼
              Red / NIR / NDVI                 Evidence display
                       │
                       ▼
              Downloadable GeoTIFF

User feedback + run logs → DuckDB/dlt → Streamlit monitoring
```

## Retrieval approaches

GeoScope implements and evaluates four real pipelines:

| Approach | Flow |
|---|---|
| Vector search | Original query → Chroma |
| Query rewriting | Rewritten query → Chroma |
| Reranking | Original query → Chroma candidates → FlashRank |
| Full pipeline | Rewritten query → Chroma candidates → FlashRank |

The evaluation page runs the same ground-truth questions through all
four pipelines and compares Hit Rate and MRR.

## Project structure

```text
GeoScope_Agent/
├── GeoScope.py
├── pages/
│   ├── 1_Data_Preparation.py
│   ├── 2_AOI_and_STAC.py
│   ├── 3_Ask_GeoAI.py
│   ├── 4_Evaluation_and_Feedback.py
│   ├── 5_Monitoring.py
│   └── 6_Automated_Demo.py
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
│   └── ui.py
├── data/
│   ├── vector_store/
│   ├── flashrank_cache/
│   ├── evaluation_questions.csv
│   └── demo/
├── docs/
│   └── USER_GUIDE.md
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.11 recommended
- Ollama installed and running
- Internet access for STAC and optional place search
- A local FlashRank model cache when Hugging Face is blocked by a
  corporate certificate

Recommended Ollama models:

```cmd
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

## Installation

```cmd
git clone <YOUR_REPOSITORY_URL>
cd GeoScope_Agent

py -3.11 -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## FlashRank model

The project uses:

```text
ms-marco-MiniLM-L-12-v2
```

The expected local structure is:

```text
data/
└── flashrank_cache/
    └── ms-marco-MiniLM-L-12-v2/
        ├── config.json
        ├── flashrank-MiniLM-L-12-v2_Q.onnx
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── tokenizer.json
```

Do not commit the model binaries unless the course or repository policy
explicitly allows it. A practical `.gitignore` entry is:

```gitignore
data/flashrank_cache/
```

## Provider configuration

Create:

```text
.streamlit/secrets.toml
```

For Ollama:

```toml
GEOSCOPE_PROVIDER = "ollama"
OLLAMA_GENERATION_MODEL = "qwen2.5:7b-instruct"
OLLAMA_JUDGE_MODEL = "llama3.1:8b"
```

For optional OpenAI reviewer mode:

```toml
GEOSCOPE_PROVIDER = "openai"
OPENAI_API_KEY = "your-key"
OPENAI_GENERATION_MODEL = "gpt-5-mini"
OPENAI_JUDGE_MODEL = "gpt-5-mini"
```

Never commit `secrets.toml`.

## Run the application

Make sure Ollama is running, then:

```cmd
python -m streamlit run GeoScope.py
```

## Quick start

1. Open **Data Preparation**.
2. Upload or confirm the PDF/HTML knowledge documents.
3. Run ingestion and build the vector index.
4. Open **AOI and STAC**.
5. Draw an AOI or search `Kom Ombo, Aswan, Egypt`.
6. Select a historical date range and search Sentinel-2.
7. Open **Ask GeoAI**.
8. Select the full retrieval pipeline.
9. Ask a technical remote-sensing question.
10. Inspect the rewritten query, vector ranks, and reranking scores.
11. Open **Evaluation and Feedback** to compare retrieval pipelines.
12. Open **Monitoring** to inspect runs and feedback.

## Automated demonstration

Open:

```text
6_Automated_Demo.py
```

The standard demo performs:

```text
AOI resolution
→ STAC search
→ distinct-date check
→ query rewriting
→ vector retrieval
→ FlashRank reranking
→ grounded answer
→ optional GeoTIFF
→ monitoring log
```

For a fast recorded demo, keep GeoTIFF generation disabled. For the
full live demo, enable it after confirming that remote raster access
works on the current network.

## Important temporal rule

```text
Number of STAC scene items ≠ number of acquisition dates
```

Several scene items can be tiles from the same day. GeoScope counts
distinct dates and does not recommend time-series analysis unless at
least two acquisition dates are available.

## GeoTIFF workflow

The current raster-processing scope is:

```text
one AOI + one STAC item + one product
→ one clipped GeoTIFF
```

Available products:

- Red
- NIR
- NDVI

The lowest-cloud returned scene can be selected automatically in the
demo, or a scene can be selected manually from the AOI/STAC page.

## Evaluation

### Retrieval

- Ground-truth questions: `data/evaluation_questions.csv`
- Metrics: Hit Rate and Mean Reciprocal Rank
- Compared approaches: vector, rewriting, reranking, full pipeline

### Generation

- LLM-as-a-judge
- relevance
- groundedness
- completeness
- technical correctness
- citation quality
- geographic relevance

### Human feedback

- thumbs up/down
- optional written comment
- logged for monitoring

## Monitoring

GeoScope records application runs and feedback in DuckDB through the
project logging layer. The Streamlit monitoring page provides metrics,
filters, history, and export.

## Corporate-network SSL limitation

Some corporate networks inject a self-signed certificate into HTTPS
connections. This can affect:

- Nominatim text geocoding;
- remote Sentinel-2 GeoTIFF access;
- first-time FlashRank downloads.

The preferred production solution is to configure the organization's
trusted root CA. Temporary insecure SSL workarounds should remain local
and must not be presented as a production security configuration.

## Current limitations

- no multi-tile mosaicking;
- no cloud-mask raster workflow yet;
- no aligned multi-date raster cube;
- no scheduled ingestion orchestration;
- no Postgres/Grafana deployment;
- no Docker deployment yet;
- automated GeoTIFF generation depends on network access;
- query rewriting increases evaluation runtime.

## Rubric self-check

| Criterion | Implementation |
|---|---|
| Problem description | `README.md`, Problem section |
| Retrieval flow | `src/retrieval.py`, `src/query_rewrite.py`, `src/reranking.py`, `src/generation.py` |
| Retrieval evaluation | `src/evaluation.py`, Evaluation page |
| LLM evaluation | LLM judge plus human feedback |
| Interface | Streamlit multipage app |
| Ingestion pipeline | Data Preparation page and ingestion modules |
| Monitoring | DuckDB/dlt and Streamlit monitoring dashboard |
| Reproducibility | README, user guide, requirements, provider examples |
| Query rewriting | `src/query_rewrite.py` |
| Re-ranking | `src/reranking.py` |
| Multiple retrieval approaches | Four pipelines compared using the same ground truth |
| Domain integration | AOI, STAC, distinct dates, NDVI, GeoTIFF |

## Security

The following files and directories must remain outside Git:

```gitignore
.venv/
__pycache__/
*.pyc
.streamlit/secrets.toml
.env
logs/*.duckdb
data/vector_store/
data/flashrank_cache/
```

## Detailed usage

See [docs/USER_GUIDE.md](documentation/USER_GUIDE.md).
