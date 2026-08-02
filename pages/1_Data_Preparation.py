from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ui import apply_global_style


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_vector_index import build_vector_index
from src.ingest_documents import ingest_documents


RAW_DIR = PROJECT_ROOT / "docs" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "docs" / "processed"
METADATA_PATH = PROCESSED_DIR / "document_metadata.csv"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="Data Preparation",
    page_icon="📚",
    layout="wide",
)

apply_global_style()

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}
.upload-card {
    padding: 1.3rem;
    border-radius: 18px;
    border: 1px solid rgba(120,120,120,0.20);
    background: rgba(255,255,255,0.025);
    margin-bottom: 1rem;
}
.flow-step {
    padding: 0.9rem 1rem;
    border-radius: 14px;
    border: 1px solid rgba(120,120,120,0.18);
    min-height: 150px;
}
.stButton > button {
    border-radius: 12px;
    min-height: 2.7rem;
    font-weight: 650;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📚 Step 1 — Data Preparation")
st.caption(
    "Upload documents, inspect the source library, ingest content, "
    "and build the vector knowledge base."
)

st.markdown(
    """
### What is implemented on this page?

This page prepares the knowledge used by GeoScope:

1. Upload or place documents in `docs/raw`.
2. Extract text from PDF, HTML, and HTM files.
3. Split the text into overlapping chunks.
4. Generate embeddings locally with `nomic-embed-text`.
5. Store the vectors in Chroma for semantic retrieval.
"""
)

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown(
        """
<div class="flow-step">
<h3>1. Source files</h3>
<p>PDF, HTML, and HTM documents are stored in <code>docs/raw</code>.</p>
</div>
""",
        unsafe_allow_html=True,
    )

with f2:
    st.markdown(
        """
<div class="flow-step">
<h3>2. Processing</h3>
<p>Text is extracted, cleaned, chunked, and recorded with document metadata.</p>
</div>
""",
        unsafe_allow_html=True,
    )

with f3:
    st.markdown(
        """
<div class="flow-step">
<h3>3. Vector index</h3>
<p>Chunks are embedded locally and added to the persistent Chroma collection.</p>
</div>
""",
        unsafe_allow_html=True,
    )

st.divider()
st.subheader("Upload documents")

st.info(
    "Supported formats: PDF, HTML, and HTM. "
    "Uploaded files are copied into docs/raw and are not indexed until "
    "you run ingestion and rebuild the vector index."
)

uploaded_files = st.file_uploader(
    "Choose one or more documents",
    type=["pdf", "html", "htm"],
    accept_multiple_files=True,
)

overwrite_existing = st.checkbox(
    "Replace files with the same name",
    value=False,
)

if uploaded_files:
    preview_rows = [
        {
            "file_name": uploaded.name,
            "size_kb": round(uploaded.size / 1024, 1),
            "type": uploaded.type or Path(uploaded.name).suffix,
        }
        for uploaded in uploaded_files
    ]

    st.dataframe(
        pd.DataFrame(preview_rows),
        use_container_width=True,
        hide_index=True,
    )

    if st.button(
        "Add uploaded documents to the source library",
        type="primary",
        use_container_width=True,
    ):
        saved = []
        skipped = []

        for uploaded in uploaded_files:
            safe_name = Path(uploaded.name).name
            destination = RAW_DIR / safe_name

            if destination.exists() and not overwrite_existing:
                skipped.append(safe_name)
                continue

            with destination.open("wb") as output:
                shutil.copyfileobj(uploaded, output)

            saved.append(safe_name)

        if saved:
            st.success(
                f"Added {len(saved)} document(s) to docs/raw."
            )
            st.write(saved)

        if skipped:
            st.warning(
                "These files already existed and were skipped. "
                "Enable replacement to overwrite them."
            )
            st.write(skipped)

st.divider()
st.subheader("Current source library")

source_files = sorted(
    [
        path
        for path in RAW_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".pdf", ".html", ".htm"}
    ]
)

if source_files:
    source_table = pd.DataFrame(
        [
            {
                "file_name": path.name,
                "type": path.suffix.lower().lstrip("."),
                "size_kb": round(path.stat().st_size / 1024, 1),
            }
            for path in source_files
        ]
    )

    st.dataframe(
        source_table,
        use_container_width=True,
        hide_index=True,
    )
    st.metric("Source documents", len(source_files))
else:
    st.warning("No supported documents are currently available in docs/raw.")

st.divider()
st.subheader("Run the preparation pipeline")

left, right = st.columns(2)

with left:
    st.markdown("### A. Ingest and chunk")

    st.write(
        "Extract text, create chunks, and update the document metadata file."
    )

    if st.button(
        "Run document ingestion",
        use_container_width=True,
    ):
        try:
            with st.spinner("Extracting and chunking documents..."):
                result = ingest_documents()

            st.success(
                f"Processed {result['documents_processed']} documents and "
                f"created {result['chunks_created']} chunks."
            )

            st.session_state["last_ingestion_result"] = result

        except Exception as exc:
            st.error(str(exc))

with right:
    st.markdown("### B. Build vector index")

    st.write(
        "Generate embeddings with Ollama and update the persistent Chroma index."
    )

    if st.button(
        "Build or update vector index",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Embedding document chunks with nomic-embed-text..."
            ):
                result = build_vector_index()

            st.success(
                f"Indexed {result['records']} chunks in "
                f"{result['collection']}."
            )

            st.session_state["last_index_result"] = result

        except Exception as exc:
            st.error(str(exc))

st.divider()
st.subheader("Preparation status")

s1, s2, s3 = st.columns(3)

with s1:
    st.metric(
        "Raw documents",
        len(source_files),
    )

with s2:
    chunk_count = 0

    if CHUNKS_PATH.exists():
        with CHUNKS_PATH.open("r", encoding="utf-8") as file:
            chunk_count = sum(1 for line in file if line.strip())

    st.metric(
        "Processed chunks",
        chunk_count,
    )

with s3:
    vector_status = "Ready" if VECTOR_DIR.exists() else "Not built"
    st.metric(
        "Vector store",
        vector_status,
    )

if METADATA_PATH.exists():
    st.markdown("### Document ingestion results")

    metadata = pd.read_csv(METADATA_PATH)

    st.dataframe(
        metadata,
        use_container_width=True,
        hide_index=True,
    )

    failed = metadata[
        metadata["status"].astype(str) != "success"
    ]

    if not failed.empty:
        st.warning(
            f"{len(failed)} document(s) require review because text "
            "could not be extracted or processing failed."
        )
