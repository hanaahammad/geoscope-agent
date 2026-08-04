from __future__ import annotations

from src.llm_provider import generate_text, get_generation_model


REWRITE_INSTRUCTIONS = """
You rewrite user questions for semantic retrieval over a remote-sensing
knowledge base.

Return one concise search query only. Preserve the scientific meaning,
dataset names, sensor names, crop names, geographic constraints, and
time constraints. Add useful remote-sensing terminology only when it
clarifies the intent. Do not answer the question. Do not add markdown,
labels, quotation marks, or explanations.
""".strip()


def rewrite_query(question: str) -> str:
    """
    Rewrite a user question into a retrieval-oriented search query.
    """
    cleaned = " ".join(question.split())

    if not cleaned:
        raise ValueError("A non-empty question is required.")

    prompt = f"""
Original user question:
{cleaned}

Rewrite it as one retrieval query for a technical remote-sensing
knowledge base.
""".strip()

    rewritten = generate_text(
        instructions=REWRITE_INSTRUCTIONS,
        prompt=prompt,
        model=get_generation_model(),
    ).strip()

    rewritten = rewritten.strip("`").strip().strip('"').strip("'")

    if not rewritten:
        return cleaned

    # Keep rewriting bounded so accidental verbose model output does not
    # become the embedding query.
    words = rewritten.split()
    return " ".join(words[:80])
