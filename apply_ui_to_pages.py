from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

targets = [PROJECT_ROOT / "app.py"]
targets.extend(sorted((PROJECT_ROOT / "pages").glob("*.py")))

import_line = "from src.ui import apply_global_style"
call_line = "apply_global_style()"

for path in targets:
    if not path.exists():
        continue

    text = path.read_text(encoding="utf-8")

    if import_line not in text:
        marker = "import streamlit as st"
        if marker not in text:
            print(f"Skipped {path.name}: Streamlit import not found.")
            continue
        text = text.replace(
            marker,
            marker + "\n\n" + import_line,
            1,
        )

    if call_line not in text:
        marker = "st.set_page_config("
        start = text.find(marker)

        if start == -1:
            print(f"Skipped {path.name}: set_page_config not found.")
            continue

        depth = 0
        end = None

        for index in range(start, len(text)):
            char = text[index]

            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1

                if depth == 0:
                    end = index + 1
                    break

        if end is None:
            print(f"Skipped {path.name}: could not parse set_page_config.")
            continue

        text = (
            text[:end]
            + "\n\n"
            + call_line
            + text[end:]
        )

    path.write_text(text, encoding="utf-8")
    print(f"Updated {path.relative_to(PROJECT_ROOT)}")
