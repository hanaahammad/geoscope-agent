# GeoScope Agent — User Guide

## 1. Start GeoScope

Open a terminal in the project directory:

```cmd
.venv\Scripts\activate
python -m streamlit run GeoScope.py
```

Confirm that Ollama is running before asking questions.

## 2. Page 1 — Data Preparation

Purpose:

- add PDF or HTML technical references;
- extract and clean text;
- split content into chunks;
- create embeddings;
- build the persistent Chroma index.

Recommended sequence:

1. Open **Data Preparation**.
2. Confirm that the source documents are available.
3. Run document ingestion.
4. Build or rebuild the vector index.
5. Confirm that chunks were stored successfully.

Rebuild the index whenever the document collection changes.

## 3. Page 2 — AOI, STAC, and GeoTIFF

### Define the AOI

Two modes are available:

- draw a rectangle or polygon;
- search a place by text.

Example:

```text
Kom Ombo, Aswan, Egypt
```

When using text search:

1. enter the place;
2. click **Search place**;
3. inspect the returned name and map;
4. click **Use this place as the AOI**.

### Search Sentinel-2

1. choose start and end dates;
2. choose maximum cloud cover;
3. choose the maximum number of scene items;
4. click **Search Sentinel-2**.

Review:

- number of scene items;
- number of distinct dates;
- cloud cover;
- acquisition dates.

Several scene items from the same date do not form a time series.

### Generate GeoTIFF

1. select one scene;
2. select Red, NIR, or NDVI;
3. click **Generate GeoTIFF**;
4. review raster dimensions, CRS, and statistics;
5. click **Download GeoTIFF**.

## 4. Page 3 — Ask GeoAI

Complete the filters, then select a retrieval approach.

### Retrieval choices

**Vector search**

```text
original question → Chroma
```

**Query rewriting**

```text
original question → LLM rewrite → Chroma
```

**Reranking**

```text
original question → Chroma candidates → FlashRank
```

**Full pipeline**

```text
original question
→ LLM rewrite
→ Chroma candidates
→ FlashRank
→ final context
```

The full pipeline is recommended for the demonstration.

Example question:

```text
Which Sentinel-2 bands and processing workflow should I use to assess
wheat vegetation condition in this area, and is the available temporal
coverage sufficient for a time series?
```

After running:

- inspect the original retrieval input;
- inspect the rewritten query;
- compare vector rank and final rank;
- review reranking scores;
- open each evidence chunk;
- verify that the answer respects the distinct-date rule.

## 5. Page 4 — Evaluation and Feedback

### Retrieval comparison

1. select domain and difficulty;
2. choose final top-k;
3. choose the candidate count;
4. click **Compare all retrieval approaches**.

The result compares:

- questions evaluated;
- Hit Rate;
- MRR;
- failures;
- question-level details.

A higher MRR means the expected source tends to appear earlier.

### Generation evaluation

1. first generate an answer in Ask GeoAI;
2. return to Evaluation;
3. run generation evaluation;
4. review judge scores and comments.

### Human feedback

Select thumbs up or thumbs down and optionally explain the rating.

## 6. Page 5 — Monitoring

Use this page to review:

- successful and failed runs;
- latency;
- applications and filters;
- questions and answers;
- feedback;
- evaluation records;
- exported CSV data.

## 7. Page 6 — Automated Demo

The automated page is designed for a presentation or screen recording.

### Standard demo

Recommended settings:

```text
Place: Kom Ombo, Aswan, Egypt
Period: 2025-11-01 to 2026-03-31
Cloud cover: 40%
Scenes: 10
Retrieval: Full pipeline
Final chunks: 5
Candidates: 15
GeoTIFF: disabled
```

Click:

```text
Run automated demonstration
```

The page then executes and displays:

1. AOI resolution;
2. STAC search;
3. distinct-date validation;
4. query rewriting;
5. semantic candidate retrieval;
6. FlashRank reranking;
7. grounded answer generation;
8. monitoring logging.

### Full live demo

Enable **Include real GeoTIFF generation**.

The page selects the lowest-cloud returned scene and generates the
chosen product. This step is slower and depends on remote raster access.

### Presentation fallback

When text geocoding fails, the page uses a bundled Kom Ombo
demonstration bounding box. This fallback is not an official
administrative boundary.

When GeoTIFF access is unreliable, run the standard demo without raster
generation, then show the manually generated file from Page 2.

## 8. Suggested 6-minute presentation script

### 0:00–0:40 — Problem

“GeoScope connects technical remote-sensing guidance with a real Area
of Interest and actual Sentinel-2 availability.”

### 0:40–1:20 — Architecture

Show the architecture in the README or Automated Demo page.

### 1:20–3:30 — Automated workflow

Run the automated demonstration and narrate each completed stage.

### 3:30–4:20 — Retrieval transparency

Show the rewritten query, original vector ranks, final ranks, and
FlashRank scores.

### 4:20–5:10 — Evaluation

Show the comparison of the four retrieval pipelines using Hit Rate and
MRR.

### 5:10–5:45 — GeoTIFF

Show a generated NDVI GeoTIFF and the download button.

### 5:45–6:00 — Conclusion

“GeoScope is not only a chatbot. It combines RAG, geospatial context,
live satellite metadata, executable raster processing, evaluation, and
monitoring.”

## 9. Troubleshooting

### Ollama connection error

Check:

```cmd
ollama list
```

Start Ollama and confirm that the configured models exist.

### FlashRank tries to download

Confirm:

```text
data\flashrank_cache\ms-marco-MiniLM-L-12-v2
```

contains the ONNX and tokenizer files.

### SSL certificate errors

These usually come from a corporate HTTPS-inspection proxy. The proper
solution is to install or configure the organization's trusted root CA.

### No STAC results

- widen the date range;
- increase maximum cloud cover;
- verify the AOI;
- reduce the requested geographic extent.

### No time series

Check the distinct-date count. Several tiles from one date are still
one temporal observation.
