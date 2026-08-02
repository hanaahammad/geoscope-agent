py -3.11 -m venv .venv
py -m pip install -r requirements.txt
py -m pip install pypdf pandas


python src\ingest_documents.py
py src\ingest_documents.py

Inside the activated environment, both py and python should work, but using py -m pip helps ensure pip is linked to the intended interpreter.
