from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "docs" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "docs" / "processed"

CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
METADATA_FILE = PROCESSED_DIR / "document_metadata.csv"

CHUNK_SIZE = 1_200
CHUNK_OVERLAP = 200


def clean_text(text: str) -> str:
    """Normalize extracted PDF or HTML text."""
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if end < len(text):
            sentence_break = max(
                chunk.rfind(". "),
                chunk.rfind("? "),
                chunk.rfind("! "),
            )

            if sentence_break > chunk_size // 2:
                end = start + sentence_break + 1
                chunk = text[start:end]

        chunk = chunk.strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def make_document_id(path: Path) -> str:
    """Create a stable identifier from a file name."""
    document_id = path.stem.lower()
    return re.sub(r"[^a-z0-9]+", "_", document_id).strip("_")


def process_pdf(
    pdf_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract and chunk one PDF."""
    reader = PdfReader(pdf_path)

    records: list[dict[str, Any]] = []
    extracted_pages = 0
    empty_pages = 0
    document_id = make_document_id(pdf_path)

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")

        if not text:
            empty_pages += 1
            continue

        extracted_pages += 1

        for chunk_number, chunk in enumerate(
            split_text(text),
            start=1,
        ):
            records.append(
                {
                    "chunk_id": (
                        f"{document_id}"
                        f"_p{page_number:04d}"
                        f"_c{chunk_number:03d}"
                    ),
                    "document_id": document_id,
                    "file_name": pdf_path.name,
                    "page_number": page_number,
                    "chunk_number": chunk_number,
                    "text": chunk,
                    "title": pdf_path.stem,
                    "source_type": "pdf",
                }
            )

    metadata = {
        "document_id": document_id,
        "file_name": pdf_path.name,
        "source_type": "pdf",
        "total_pages": len(reader.pages),
        "pages_with_text": extracted_pages,
        "empty_pages": empty_pages,
        "number_of_chunks": len(records),
        "status": "success" if records else "no_text_extracted",
    }

    return records, metadata


def process_html(
    html_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract and chunk one saved HTML or HTM page."""
    raw_html = html_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    soup = BeautifulSoup(raw_html, "lxml")

    for element in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "noscript",
            "svg",
        ]
    ):
        element.decompose()

    title = ""

    if soup.title and soup.title.string:
        title = clean_text(soup.title.string)

    content_root = (
        soup.find("article")
        or soup.find("main")
        or soup.body
        or soup
    )

    content_parts: list[str] = []

    for element in content_root.find_all(
        ["h1", "h2", "h3", "h4", "p", "li"]
    ):
        text = clean_text(
            element.get_text(" ", strip=True)
        )

        if len(text) >= 20:
            content_parts.append(text)

    full_text = clean_text("\n".join(content_parts))
    document_id = make_document_id(html_path)

    records: list[dict[str, Any]] = []

    for chunk_number, chunk in enumerate(
        split_text(full_text),
        start=1,
    ):
        records.append(
            {
                "chunk_id": (
                    f"{document_id}"
                    f"_html_c{chunk_number:03d}"
                ),
                "document_id": document_id,
                "file_name": html_path.name,
                "page_number": 1,
                "chunk_number": chunk_number,
                "text": chunk,
                "title": title or html_path.stem,
                "source_type": "html",
            }
        )

    metadata = {
        "document_id": document_id,
        "file_name": html_path.name,
        "source_type": "html",
        "total_pages": 1,
        "pages_with_text": 1 if records else 0,
        "empty_pages": 0 if records else 1,
        "number_of_chunks": len(records),
        "status": "success" if records else "no_text_extracted",
    }

    return records, metadata


def ingest_documents() -> dict[str, Any]:
    """
    Ingest all PDF, HTML, and HTM documents from docs/raw.

    This function is used by Streamlit and can also be called directly.
    It writes:
      - docs/processed/chunks.jsonl
      - docs/processed/document_metadata.csv
    """
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_files = (
        sorted(RAW_DIR.glob("*.pdf"))
        + sorted(RAW_DIR.glob("*.html"))
        + sorted(RAW_DIR.glob("*.htm"))
    )

    if not source_files:
        raise FileNotFoundError(
            f"No PDF, HTML, or HTM files were found in: {RAW_DIR}"
        )

    all_chunks: list[dict[str, Any]] = []
    all_metadata: list[dict[str, Any]] = []

    for source_path in source_files:
        try:
            if source_path.suffix.lower() == ".pdf":
                chunks, metadata = process_pdf(source_path)
            else:
                chunks, metadata = process_html(source_path)

            all_chunks.extend(chunks)
            all_metadata.append(metadata)

        except Exception as exc:
            all_metadata.append(
                {
                    "document_id": make_document_id(source_path),
                    "file_name": source_path.name,
                    "source_type": source_path.suffix.lower().lstrip("."),
                    "total_pages": "",
                    "pages_with_text": "",
                    "empty_pages": "",
                    "number_of_chunks": 0,
                    "status": f"failed: {exc}",
                }
            )

    with CHUNKS_FILE.open("w", encoding="utf-8") as file:
        for record in all_chunks:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    fieldnames = [
        "document_id",
        "file_name",
        "source_type",
        "total_pages",
        "pages_with_text",
        "empty_pages",
        "number_of_chunks",
        "status",
    ]

    with METADATA_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(all_metadata)

    return {
        "status": "success",
        "documents_processed": len(all_metadata),
        "chunks_created": len(all_chunks),
        "chunks_file": str(CHUNKS_FILE),
        "metadata_file": str(METADATA_FILE),
        "metadata": all_metadata,
    }


def main() -> None:
    """Command-line entry point."""
    result = ingest_documents()
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
