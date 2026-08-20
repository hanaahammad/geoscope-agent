from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from src.ui import apply_global_style


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Cloud Deployment Test",
    page_icon="☁️",
    layout="wide",
)

apply_global_style()

OLLAMA_CLOUD_HOST = "https://ollama.com"
OLLAMA_API_BASE = f"{OLLAMA_CLOUD_HOST}/api"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _secret(name: str, default: str = "") -> str:
    """
    Read a value from Streamlit Secrets first, then environment variables.

    This works both locally and on Streamlit Community Cloud.
    """
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except Exception:
        pass

    return os.getenv(name, default)


def _headers(api_key: str) -> dict[str, str]:
    if not api_key:
        raise ValueError(
            "OLLAMA_API_KEY is not configured. "
            "Add it to Streamlit Secrets before using Ollama Cloud."
        )

    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "GeoScope-Agent/1.0",
    }


def list_cloud_models(
    api_key: str,
) -> list[str]:
    """
    Return models currently exposed by Ollama Cloud for this account.
    """
    response = requests.get(
        f"{OLLAMA_API_BASE}/tags",
        headers=_headers(api_key),
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()

    models = []

    for item in payload.get("models", []):
        name = item.get("name") or item.get("model")
        if name:
            models.append(str(name))

    return sorted(set(models))


def cloud_chat(
    *,
    api_key: str,
    model: str,
    prompt: str,
) -> dict[str, Any]:
    """
    Send a non-streaming chat request to Ollama Cloud.
    """
    response = requests.post(
        f"{OLLAMA_API_BASE}/chat",
        headers=_headers(api_key),
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()

    return response.json()


def cloud_embed(
    *,
    api_key: str,
    model: str,
    text: str,
) -> dict[str, Any]:
    """
    Send an embedding request to Ollama Cloud.
    """
    response = requests.post(
        f"{OLLAMA_API_BASE}/embed",
        headers=_headers(api_key),
        json={
            "model": model,
            "input": text,
        },
        timeout=180,
    )
    response.raise_for_status()

    return response.json()


def get_existing_chroma_dimension() -> int | None:
    """
    Inspect one stored Chroma embedding without modifying the index.

    Returns None if the existing vector store is unavailable.
    """
    try:
        from src.retrieval import get_collection

        collection = get_collection()
        sample = collection.peek(limit=1)

        embeddings = sample.get("embeddings")

        if embeddings is None or len(embeddings) == 0:
            return None

        return len(embeddings[0])

    except Exception:
        return None


def mask_key_status(api_key: str) -> str:
    if not api_key:
        return "Not configured"

    if len(api_key) <= 8:
        return "Configured"

    return (
        f"Configured · {api_key[:4]}…{api_key[-4:]}"
    )


# ---------------------------------------------------------------------------
# Header / explanation for reviewer
# ---------------------------------------------------------------------------

st.title("☁️ GeoScope — Cloud Deployment Test")

st.markdown(
    """
This page is intentionally isolated from the existing GeoScope runtime.

It validates whether **Ollama Cloud** can be used as the remote model
provider when GeoScope is deployed on **Streamlit Community Cloud**.

```text
GitHub repository
      │
      ▼
Streamlit Community Cloud
      │
      ├── GeoScope Streamlit UI
      ├── Chroma / project assets
      │
      └──────── HTTPS ────────▶ Ollama Cloud
                                generation / judge / embeddings
```

**Important terminology for peer review:** GeoScope is *deployed on
Streamlit Community Cloud* and *uses Ollama Cloud for remote model
inference*. GeoScope itself is not deployed on Ollama Cloud.
"""
)

st.info(
    "This page does not replace the current local Ollama provider and "
    "does not modify src/llm_provider.py, retrieval.py, or generation.py."
)


# ---------------------------------------------------------------------------
# Reviewer setup
# ---------------------------------------------------------------------------

with st.expander(
    "Reviewer setup — how to enable Ollama Cloud",
    expanded=True,
):
    st.markdown(
        """
1. Create or use an Ollama account.
2. Create an Ollama API key.
3. Open the deployed Streamlit application's **Settings / Secrets**.
4. Add:

```toml
OLLAMA_API_KEY = "your_ollama_api_key"
```

Optional default model values can also be supplied:

```toml
OLLAMA_CLOUD_GENERATION_MODEL = "gpt-oss:120b"
OLLAMA_CLOUD_JUDGE_MODEL = "gpt-oss:120b"
OLLAMA_CLOUD_EMBEDDING_MODEL = "embeddinggemma"
```

5. Save the secrets and restart the Streamlit application.
6. Open this page and click **Test Ollama Cloud connection**.

The API key must never be committed to GitHub.
"""
    )


# ---------------------------------------------------------------------------
# Configuration status
# ---------------------------------------------------------------------------

api_key = _secret("OLLAMA_API_KEY")

default_generation_model = _secret(
    "OLLAMA_CLOUD_GENERATION_MODEL",
    "gpt-oss:120b",
)

default_judge_model = _secret(
    "OLLAMA_CLOUD_JUDGE_MODEL",
    default_generation_model,
)

default_embedding_model = _secret(
    "OLLAMA_CLOUD_EMBEDDING_MODEL",
    "embeddinggemma",
)


st.subheader("1. Deployment configuration")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "Application host",
        "Streamlit Cloud",
    )

with m2:
    st.metric(
        "Remote LLM provider",
        "Ollama Cloud",
    )

with m3:
    st.metric(
        "API authentication",
        "Configured" if api_key else "Missing",
    )

st.caption(
    f"OLLAMA_API_KEY: {mask_key_status(api_key)}"
)


# ---------------------------------------------------------------------------
# Connection and model discovery
# ---------------------------------------------------------------------------

st.subheader("2. Test Ollama Cloud connection")

if st.button(
    "Test Ollama Cloud connection",
    type="primary",
    use_container_width=True,
):
    try:
        started = time.perf_counter()

        models = list_cloud_models(api_key)

        latency = time.perf_counter() - started

        st.session_state[
            "ollama_cloud_models"
        ] = models

        st.session_state[
            "ollama_cloud_connection_latency"
        ] = latency

        st.success(
            f"Connection successful. "
            f"{len(models)} model(s) returned in "
            f"{latency:.2f} seconds."
        )

    except Exception as exc:
        st.error(
            "Ollama Cloud connection failed. "
            f"Technical detail: {exc}"
        )


cloud_models = st.session_state.get(
    "ollama_cloud_models",
    [],
)

if cloud_models:
    with st.expander(
        "Models returned by Ollama Cloud"
    ):
        st.write(cloud_models)


# ---------------------------------------------------------------------------
# Generation test
# ---------------------------------------------------------------------------

st.divider()
st.subheader("3. Test cloud generation")

generation_options = (
    cloud_models
    if cloud_models
    else [default_generation_model]
)

if (
    default_generation_model not in generation_options
    and cloud_models
):
    generation_options = [
        default_generation_model,
        *generation_options,
    ]

generation_model = st.selectbox(
    "Generation model",
    generation_options,
    index=0,
)

generation_prompt = st.text_area(
    "Generation test prompt",
    value=(
        "In two short sentences, explain why Sentinel-2 is useful "
        "for vegetation monitoring."
    ),
    height=100,
)

if st.button(
    "Run generation test",
    use_container_width=True,
):
    try:
        started = time.perf_counter()

        result = cloud_chat(
            api_key=api_key,
            model=generation_model,
            prompt=generation_prompt,
        )

        elapsed = time.perf_counter() - started

        answer = (
            result.get("message", {})
            .get("content", "")
        )

        st.session_state[
            "ollama_cloud_generation_result"
        ] = {
            "model": generation_model,
            "answer": answer,
            "latency": elapsed,
            "prompt_eval_count": result.get(
                "prompt_eval_count"
            ),
            "eval_count": result.get(
                "eval_count"
            ),
        }

    except Exception as exc:
        st.error(
            "Cloud generation failed. "
            f"Technical detail: {exc}"
        )


generation_result = st.session_state.get(
    "ollama_cloud_generation_result"
)

if generation_result:
    st.success("Cloud generation successful.")

    g1, g2, g3 = st.columns(3)

    with g1:
        st.metric(
            "Model",
            generation_result["model"],
        )

    with g2:
        st.metric(
            "Latency",
            f"{generation_result['latency']:.2f}s",
        )

    with g3:
        st.metric(
            "Generated tokens",
            generation_result.get(
                "eval_count"
            )
            or "N/A",
        )

    st.markdown("**Response**")
    st.write(
        generation_result["answer"]
    )


# ---------------------------------------------------------------------------
# Query rewrite / judge suitability
# ---------------------------------------------------------------------------

st.divider()
st.subheader("4. Test GeoScope LLM tasks")

task = st.radio(
    "Task",
    [
        "Query rewriting",
        "LLM judge",
    ],
    horizontal=True,
)

if task == "Query rewriting":
    task_prompt = st.text_area(
        "Query rewrite test",
        value=(
            "Rewrite this for remote-sensing semantic retrieval. "
            "Return only the rewritten query: "
            "'How can I check wheat condition in Kom Ombo?'"
        ),
        height=100,
    )
    task_model = generation_model

else:
    judge_options = (
        cloud_models
        if cloud_models
        else [default_judge_model]
    )

    task_model = st.selectbox(
        "Judge model",
        judge_options,
    )

    task_prompt = st.text_area(
        "Judge test",
        value=(
            "Return JSON only with relevance and groundedness scores "
            "from 1 to 5. Question: Which Sentinel-2 bands are used "
            "for NDVI? Answer: NDVI uses Red and NIR bands."
        ),
        height=110,
    )

if st.button(
    "Run task test",
    use_container_width=True,
):
    try:
        result = cloud_chat(
            api_key=api_key,
            model=task_model,
            prompt=task_prompt,
        )

        st.success(
            f"{task} test successful."
        )

        st.write(
            result.get(
                "message",
                {},
            ).get(
                "content",
                "",
            )
        )

    except Exception as exc:
        st.error(
            f"{task} test failed. "
            f"Technical detail: {exc}"
        )


# ---------------------------------------------------------------------------
# Embedding compatibility test
# ---------------------------------------------------------------------------

st.divider()
st.subheader("5. Test cloud embeddings")

st.warning(
    "GeoScope must use the same embedding model for indexing and "
    "querying. Matching vector dimensions alone is necessary but is "
    "not sufficient to prove semantic compatibility."
)

embedding_model = st.text_input(
    "Cloud embedding model",
    value=default_embedding_model,
    help=(
        "The model must support Ollama's /api/embed endpoint and must "
        "match the model used to build the vector index if the existing "
        "Chroma index is to be reused."
    ),
)

embedding_text = st.text_input(
    "Embedding test text",
    value="Sentinel-2 vegetation monitoring with NDVI",
)

existing_dimension = get_existing_chroma_dimension()

if existing_dimension:
    st.write(
        f"Existing Chroma vector dimension: "
        f"**{existing_dimension}**"
    )
else:
    st.caption(
        "Existing Chroma embedding dimension could not be read "
        "from this environment."
    )

if st.button(
    "Run embedding test",
    use_container_width=True,
):
    try:
        started = time.perf_counter()

        embedding_result = cloud_embed(
            api_key=api_key,
            model=embedding_model,
            text=embedding_text,
        )

        elapsed = time.perf_counter() - started

        embeddings = embedding_result.get(
            "embeddings",
            [],
        )

        if not embeddings:
            raise RuntimeError(
                "Ollama returned no embeddings."
            )

        dimension = len(
            embeddings[0]
        )

        st.session_state[
            "ollama_cloud_embedding_test"
        ] = {
            "model": embedding_model,
            "dimension": dimension,
            "latency": elapsed,
        }

    except Exception as exc:
        st.error(
            "Cloud embedding test failed. "
            f"Technical detail: {exc}"
        )


embedding_test = st.session_state.get(
    "ollama_cloud_embedding_test"
)

if embedding_test:
    e1, e2, e3 = st.columns(3)

    with e1:
        st.metric(
            "Embedding model",
            embedding_test["model"],
        )

    with e2:
        st.metric(
            "Vector dimension",
            embedding_test["dimension"],
        )

    with e3:
        st.metric(
            "Latency",
            f"{embedding_test['latency']:.2f}s",
        )

    if existing_dimension:
        if (
            embedding_test["dimension"]
            == existing_dimension
        ):
            st.info(
                "The cloud vector dimension matches the existing "
                "Chroma dimension. This does NOT by itself prove that "
                "the index can be reused. Reuse is safe only when the "
                "same embedding model/configuration was used for both "
                "indexing and querying."
            )

        else:
            st.error(
                "The cloud embedding dimension differs from the "
                "existing Chroma dimension. The existing index cannot "
                "be queried with this embedding model and would need "
                "to be rebuilt."
            )


# ---------------------------------------------------------------------------
# Reviewer summary
# ---------------------------------------------------------------------------

st.divider()
st.subheader("6. Deployment readiness summary")

connection_ok = bool(
    st.session_state.get(
        "ollama_cloud_models"
    )
)

generation_ok = bool(
    st.session_state.get(
        "ollama_cloud_generation_result"
    )
)

embedding_ok = bool(
    st.session_state.get(
        "ollama_cloud_embedding_test"
    )
)

summary_rows = [
    {
        "Check": "Ollama Cloud authentication",
        "Status": (
            "Ready" if api_key else "Missing"
        ),
        "Purpose": "Authenticated remote model access",
    },
    {
        "Check": "Cloud API connection",
        "Status": (
            "Passed" if connection_ok else "Not tested"
        ),
        "Purpose": "Reach https://ollama.com/api",
    },
    {
        "Check": "Generation",
        "Status": (
            "Passed" if generation_ok else "Not tested"
        ),
        "Purpose": "GeoScope answer / rewrite / judge",
    },
    {
        "Check": "Embeddings",
        "Status": (
            "Passed" if embedding_ok else "Not tested"
        ),
        "Purpose": "RAG query embeddings",
    },
]

st.dataframe(
    summary_rows,
    use_container_width=True,
    hide_index=True,
)

if (
    api_key
    and connection_ok
    and generation_ok
):
    st.success(
        "Remote LLM inference is ready for a Streamlit Cloud "
        "deployment test. Validate embedding compatibility before "
        "switching the existing RAG pipeline to cloud embeddings."
    )
else:
    st.info(
        "Complete the checks above before integrating Ollama Cloud "
        "with the production GeoScope provider layer."
    )


# ---------------------------------------------------------------------------
# README-ready reviewer note
# ---------------------------------------------------------------------------

with st.expander(
    "Text to include in README / peer-review instructions"
):
    st.code(
        """
Cloud deployment option

GeoScope can run locally with Ollama or be deployed on Streamlit
Community Cloud while using Ollama Cloud for remote model inference.

For peer review:
1. Create an Ollama API key.
2. Add OLLAMA_API_KEY to the Streamlit application's Secrets.
3. Open the Cloud Deployment Test page.
4. Test the Ollama Cloud connection.
5. Test generation and query rewriting.
6. Test the selected embedding model before reusing the existing
   Chroma vector index.

Secrets are never committed to the repository.
""".strip(),
        language="text",
    )
