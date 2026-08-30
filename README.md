### FlashRank availability and reviewer safety

GeoScope uses the FlashRank reranker:

`ms-marco-MiniLM-L-12-v2`

The model cache is intentionally excluded from Git. When a reviewer selects a
reranking approach:

- GeoScope checks whether the local FlashRank cache is present;
- if it is missing, GeoScope can still try FlashRank's normal first-time model initialization/download;
- if initialization is not possible, the application **does not crash with an unhandled traceback**;
- instead, it displays a clear message and asks the reviewer to use **Vector search** or **Query rewriting + vector search**, or to install/cache the reranker and retry.

This behavior keeps the retrieval method transparent: GeoScope never silently
claims that reranking was executed when the reranker was unavailable.
