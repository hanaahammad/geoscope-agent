from __future__ import annotations

import streamlit as st


def apply_global_style() -> None:
    """Apply the shared GeoScope sage-green visual theme."""
    st.markdown(
        """
<style>
:root {
    --gs-sage-900: #24332B;
    --gs-sage-800: #31473A;
    --gs-sage-700: #4F7D68;
    --gs-sage-100: #EAF2E8;
    --gs-sage-050: #F4F8F3;
    --gs-border: rgba(79, 125, 104, 0.22);
    --gs-muted: #627267;
}

html,
body,
[data-testid="stAppViewContainer"] {
    background: var(--gs-sage-050);
    color: var(--gs-sage-900);
}

[data-testid="stSidebar"] {
    background: var(--gs-sage-100);
    border-right: 1px solid var(--gs-border);
}

.gs-provider-box {
    padding: 0.8rem 0.85rem;
    margin: 0.65rem 0 1rem 0;
    border-radius: 14px;
    border: 1px solid rgba(79, 125, 104, 0.28);
    background: rgba(255, 255, 255, 0.72);
}

.gs-provider-label {
    color: var(--gs-muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 700;
}

.gs-provider-value {
    color: var(--gs-sage-900);
    font-weight: 700;
    margin-top: 0.15rem;
}

.gs-provider-model {
    color: var(--gs-muted);
    font-size: 0.82rem;
    margin-top: 0.15rem;
}
</style>
""",
        unsafe_allow_html=True,
    )

    render_provider_status()


def render_provider_status() -> None:
    """Show the active LLM provider and models in the sidebar."""
    try:
        from src.llm_provider import (
            get_generation_model,
            get_judge_model,
            get_provider,
        )

        provider = get_provider()
        generation_model = get_generation_model()
        judge_model = get_judge_model()

        provider_label = "OpenAI" if provider == "openai" else "Local Ollama"
        icon = "☁️" if provider == "openai" else "💻"

        with st.sidebar:
            st.markdown(
                f"""
<div class="gs-provider-box">
    <div class="gs-provider-label">Active AI provider</div>
    <div class="gs-provider-value">{icon} {provider_label}</div>
    <div class="gs-provider-model">
        Generator: {generation_model}<br>
        Judge: {judge_model}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

    except Exception as exc:
        with st.sidebar:
            st.warning(f"Provider configuration could not be loaded: {exc}")


def page_hero(
    title: str,
    description: str,
    *,
    eyebrow: str = "GeoScope Agent",
    chips: list[str] | None = None,
) -> None:
    chip_html = "".join(
        f'<span class="gs-chip">{chip}</span>'
        for chip in (chips or [])
    )

    st.markdown(
        f"""
<div class="gs-hero">
    <div class="gs-eyebrow">{eyebrow}</div>
    <h1>{title}</h1>
    <p>{description}</p>
    <div>{chip_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def explanation_card(title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="gs-card">
    <h3>{title}</h3>
    <p>{body}</p>
</div>
""",
        unsafe_allow_html=True,
    )
