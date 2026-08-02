from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from pypdf import PdfReader
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "docs" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "docs" / "processed"

CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
METADATA_FILE = PROCESSED_DIR / "document_metadata.csv"

CHUNK_SIZE = 1_200
CHUNK_OVERLAP = 200


def clean_text(text: str) -> str:
    """Normalize extracted PDF text."""
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


def process_pdf(pdf_path: Path) -> tuple[list[dict], dict]:
    """Extract and chunk one PDF."""
    reader = PdfReader(pdf_path)

    records: list[dict] = []
    extracted_pages = 0
    empty_pages = 0

    document_id = pdf_path.stem.lower()
    document_id = re.sub(r"[^a-z0-9]+", "_", document_id).strip("_")

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = clean_text(raw_text)

        if not text:
            empty_pages += 1
            continue

        extracted_pages += 1
        page_chunks = split_text(text)

        for chunk_number, chunk in enumerate(page_chunks, start=1):
            chunk_id = (
                f"{document_id}"
                f"_p{page_number:04d}"
                f"_c{chunk_number:03d}"
            )

            records.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "file_name": pdf_path.name,
                    "page_number": page_number,
                    "chunk_number": chunk_number,
                    "text": chunk,
                }
            )

    metadata = {
        "document_id": document_id,
        "file_name": pdf_path.name,
        "total_pages": len(reader.pages),
        "pages_with_text": extracted_pages,
        "empty_pages": empty_pages,
        "number_of_chunks": len(records),
        "status": "success" if records else "no_text_extracted",
    }

    return records, metadata

def process_html(html_path: Path) -> tuple[list[dict], dict]:
    """Extract and chunk one saved HTML page."""

    raw_html = html_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    soup = BeautifulSoup(raw_html, "lxml")

    # Remove elements that usually add noise.
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

    content_parts: list[str] = []

    # Prefer the main article content when available.
    content_root = (
        soup.find("article")
        or soup.find("main")
        or soup.body
        or soup
    )

    for element in content_root.find_all(
        ["h1", "h2", "h3", "h4", "p", "li"]
    ):
        text = clean_text(
            element.get_text(" ", strip=True)
        )

        if len(text) >= 20:
            content_parts.append(text)

    full_text = clean_text(
        "\n".join(content_parts)
    )

    document_id = html_path.stem.lower()
    document_id = re.sub(
        r"[^a-z0-9]+",
        "_",
        document_id,
    ).strip("_")

    records: list[dict] = []

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
                "title": title,
                "source_type": "html",
            }
        )

    metadata = {
        "document_id": document_id,
        "file_name": html_path.name,
        "total_pages": 1,
        "pages_with_text": 1 if records else 0,
        "empty_pages": 0 if records else 1,
        "number_of_chunks": len(records),
        "status": (
            "success"
            if records
            else "no_text_extracted"
        ),
    }

    return records, metadata


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    '''
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files were found in: {RAW_DIR}"
        )
    '''
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    html_files = sorted(RAW_DIR.glob("*.html"))

    source_files = pdf_files + html_files

    if not source_files:
        raise FileNotFoundError(
        f"No PDF or HTML files were found in: {RAW_DIR}"
        )

    all_chunks: list[dict] = []
    all_metadata: list[dict] = []

    print(f"Found {len(pdf_files)} PDF files.\n")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        try:
            chunks, metadata = process_pdf(pdf_path)
            all_chunks.extend(chunks)
            all_metadata.append(metadata)

            print(
                f"  Pages: {metadata['total_pages']} | "
                f"Text pages: {metadata['pages_with_text']} | "
                f"Chunks: {metadata['number_of_chunks']}"
            )

        except Exception as exc:
            print(f"  ERROR: {exc}")

            all_metadata.append(
                {
                    "document_id": pdf_path.stem,
                    "file_name": pdf_path.name,
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
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metadata)

    print("\nIngestion complete.")
    print(f"Documents processed: {len(all_metadata)}")
    print(f"Chunks created: {len(all_chunks)}")
    print(f"Chunks file: {CHUNKS_FILE}")
    print(f"Metadata file: {METADATA_FILE}")


if __name__ == "__main__":
    main()