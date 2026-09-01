from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="GeoScope Agent",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------

st.title("🌍 GeoScope Agent")
st.subheader("AI-Assisted Earth Observation — from grounded knowledge to adaptive workflows")

st.markdown(
    """
**GeoScope Agent** explores how Generative AI, Retrieval-Augmented Generation
(RAG), geospatial data, and agentic workflows can support Earth Observation
researchers and practitioners.

The application is intentionally broader than a chatbot. It demonstrates both:

- an **Earth Observation assistant**, combining technical knowledge, AOI context,
  satellite-data discovery, and selected raster-processing workflows; and
- an **AI engineering lifecycle**, where retrieval, prompts, context,
  evaluation, monitoring, human feedback, and orchestration can be inspected
  rather than hidden.

> **Design principle:** use a predictable pipeline when the required sequence
> is known, and use bounded agentic orchestration only when the next action
> genuinely depends on the current context.
"""
)

st.info(
    "👀 **Reviewer tip:** GeoScope exposes the evidence, retrieval decisions, "
    "evaluation results, workflow state, and orchestration pattern behind the "
    "answer — not only the final generated text."
)


# ---------------------------------------------------------------------------
# WHAT THE PROJECT DEMONSTRATES
# ---------------------------------------------------------------------------

st.markdown("## 🧩 What GeoScope demonstrates")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("### 📚 RAG")
    st.write(
        "Document ingestion, chunking, embeddings, Chroma retrieval, "
        "query rewriting, and reranking."
    )

with c2:
    st.markdown("### 🧠 AI Engineering")
    st.write(
        "Prompt/context construction, experiment comparison, LLM-as-a-judge, "
        "human feedback, and monitoring."
    )

with c3:
    st.markdown("### 🛰️ Earth Observation")
    st.write(
        "AOI context, STAC search, satellite-data selection, derived indices, "
        "and transparent workflow limitations."
    )

with c4:
    st.markdown("### 🔀 Orchestration")
    st.write(
        "A fixed LangChain RAG pipeline compared with a bounded "
        "LangGraph agentic workflow."
    )


# ---------------------------------------------------------------------------
# APP MAP
# ---------------------------------------------------------------------------

st.markdown("## 🗺️ Explore the application")
st.caption(
    "Each page focuses on one part of the technical or analytical lifecycle."
)

pages = [
    {
        "page": "1 · Data Preparation",
        "purpose": "Prepare the knowledge base",
        "concepts": "Ingestion · chunking · embeddings · Chroma",
    },
    {
        "page": "2 · AOI & STAC",
        "purpose": "Explore geographic and satellite-data context",
        "concepts": "AOI · STAC · scene discovery · GeoTIFF",
    },
    {
        "page": "3 · Ask GeoAI",
        "purpose": "Ask grounded Earth Observation questions",
        "concepts": "RAG · rewrite · reranking · evidence · citations",
    },
    {
        "page": "4 · Evaluation & Feedback",
        "purpose": "Measure retrieval and answer quality",
        "concepts": "Hit Rate · MRR · LLM-as-a-judge · human feedback",
    },
    {
        "page": "5 · Monitoring",
        "purpose": "Observe and diagnose the AI system",
        "concepts": "Quality · governance · prompts · context · traces",
    },
    {
        "page": "6 · Automated Demo",
        "purpose": "Run a compact guided scenario",
        "concepts": "End-to-end demonstration",
    },
    {
        "page": "7 · Projects & Workflows",
        "purpose": "Move from a research goal to an EO workflow",
        "concepts": "Context-aware assistance · processing · results · project state",
    },
    {
        "page": "8 · Cloud Deployment Test",
        "purpose": "Test an optional remote-inference deployment pattern",
        "concepts": "Cloud authentication · model access · integration test",
    },
    {
        "page": "9 · Pipeline vs Agentic",
        "purpose": "Compare predictable and adaptive orchestration",
        "concepts": "LangChain · LangGraph · traces · tool selection",
    },
]

for item in pages:
    p1, p2, p3 = st.columns([1.35, 2.0, 2.7])
    with p1:
        st.markdown(f"**{item['page']}**")
    with p2:
        st.write(item["purpose"])
    with p3:
        st.caption(item["concepts"])

st.caption(
    "Use the Streamlit navigation menu in the sidebar to open each page."
)


# ---------------------------------------------------------------------------
# REVIEW PATHS
# ---------------------------------------------------------------------------

st.markdown("## 🚀 Suggested ways to explore GeoScope")

review_tab, eo_tab = st.tabs(
    ["🎓 Quick Capstone Review", "🛰️ Earth Observation Workflow"]
)

with review_tab:
    st.markdown(
        """
A reviewer who wants to understand the **LLM / RAG engineering** can follow:

```text
1  Data Preparation
        ↓
3  Ask GeoAI
        ↓
4  Evaluation & Feedback
        ↓
5  Monitoring
        ↓
9  Pipeline vs Agentic
```

**What to inspect**

1. How documents become chunks and embeddings.
2. How retrieval changes with rewrite and reranking.
3. Which evidence is placed in the LLM context.
4. How the judge evaluates generated answers.
5. How feedback and run metadata are monitored.
6. Why GeoScope uses LangChain for predictable execution and LangGraph for
   bounded adaptive execution.
"""
    )

with eo_tab:
    st.markdown(
        """
For the **Earth Observation use case**, start with:

```text
7  Projects & Workflows
        ↓
Research goal / EO task
        ↓
AOI + dates
        ↓
Dataset recommendation
        ↓
Actual STAC availability
        ↓
Processing / indicator
        ↓
Map + statistics
        ↓
Interpretation + next action
```

Page 7 is deliberately **context-aware rather than a rigid recipe**. EO
workflows can change according to sensor, acquisition dates, available bands,
processing level, cloud conditions, and remote-data access.
"""
    )


# ---------------------------------------------------------------------------
# AI ENGINEERING VIEW
# ---------------------------------------------------------------------------

st.markdown("## 🧠 AI engineering view")

st.markdown(
    """
GeoScope is designed so that a generated answer can eventually be traced back
to the configuration that produced it:

```text
Question
   │
   ├── Prompt / prompt version
   ├── Geographic context
   ├── Retrieval strategy
   ├── Top-k / retrieved chunks
   ├── Context size
   ├── Model
   └── Orchestration
          │
          ├── LangChain fixed pipeline
          └── LangGraph bounded agent
   ↓
Generated answer
   ↓
LLM-as-a-judge
   ↓
Human feedback
   ↓
Monitoring / comparison / diagnosis
```

This supports a practical question that matters during AI development:

> **When answer quality changes, did the problem come from the model, the
> prompt, retrieval, context construction, or orchestration?**
"""
)


# ---------------------------------------------------------------------------
# FIXED VS AGENTIC
# ---------------------------------------------------------------------------

st.markdown("## 🔀 Fixed pipeline vs agentic workflow")

left, right = st.columns(2)

with left:
    st.markdown("### 🔹 Fixed — LangChain")
    st.code(
        "Question → Rewrite → Retrieve → Rerank → Context → Generate",
        language=None,
    )
    st.write(
        "Best when the required sequence is known, repeatable, and should be "
        "easy to test and reproduce."
    )

with right:
    st.markdown("### 🔸 Agentic — LangGraph")
    st.code(
        "Question → Planner → Tool → Observe → Planner → … → Answer",
        language=None,
    )
    st.write(
        "Useful when the next permitted action depends on the current "
        "question, project state, geographic context, or tool result."
    )

st.success(
    "✅ **Key takeaway:** not every AI workflow needs an agent. GeoScope uses "
    "the simplest orchestration pattern that fits the task."
)


# ---------------------------------------------------------------------------
# CURRENT SCOPE AND LIMITATIONS
# ---------------------------------------------------------------------------

st.markdown("## ⚠️ Current scope and limitations")

st.markdown(
    """
GeoScope is a **capstone / prototype**, not a production remote-sensing
processing platform.

Important current limitations include:

- no complete multi-tile mosaicking pipeline;
- no full production cloud-mask workflow;
- no aligned multi-date raster cube;
- some remote raster access depends on network/provider configuration;
- Sentinel-2 NDVI processing is the clearest executable crop-monitoring example;
- Landsat Surface Temperature processing is only partially automated while
  remote access is being validated;
- several EO tasks in Projects & Workflows are currently **guided workflows**
  rather than fully automated processors;
- the LangGraph agent is deliberately **bounded** and should not be interpreted
  as an autonomous scientific decision-maker;
- generated recommendations and EO indicators still require domain validation.

These limitations are intentionally visible so that the application does not
claim capabilities that have not been implemented or scientifically validated.
"""
)


# ---------------------------------------------------------------------------
# WHAT TO LOOK FOR
# ---------------------------------------------------------------------------

st.markdown("## 🔎 What to look for while reviewing")

st.markdown(
    """
- **Evidence before generation** — retrieved chunks and sources remain visible.
- **Advanced retrieval** — compare vector, rewrite, rerank, and rewrite + rerank.
- **Evaluation** — retrieval metrics and an independent LLM-as-a-judge.
- **Human oversight** — explicit user feedback.
- **Context awareness** — AOI, dates, STAC results, and project state can affect
  recommendations.
- **Framework usage** — LangChain and LangGraph are used for different
  orchestration needs.
- **Transparency** — GeoScope states both implemented capabilities and current
  limitations.
"""
)

st.divider()
st.caption(
    "GeoScope Agent · LLM Zoomcamp Capstone · "
    "RAG + Earth Observation + AI Evaluation + Agentic Workflows"
)
