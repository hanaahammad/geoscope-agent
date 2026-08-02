# OpenAI Reviewer Setup

GeoScope supports two providers:

- `ollama`: local generation and local LLM-as-a-judge
- `openai`: OpenAI generation and OpenAI LLM-as-a-judge

The OpenAI option is useful for reviewers who do not have the required Ollama models installed.

## Important billing note

ChatGPT subscriptions and OpenAI API usage are billed separately. A paid ChatGPT plan does not automatically include API credits.

## Security rules

- Use your own API key.
- Never paste the key into source code.
- Never commit `.streamlit/secrets.toml` to GitHub.
- Never share the key with another reviewer.
- Rotate the key if it is exposed.

## Recommended setup: Streamlit secrets

### 1. Create an OpenAI API key

1. Sign in to the OpenAI API platform.
2. Open the API Keys page.
3. Create a new secret key for this review.
4. Copy it immediately and keep it private.

### 2. Confirm API billing

Open the API billing settings and confirm that the account has available credits or an active payment method.

### 3. Install the OpenAI SDK

```powershell
cd C:\path\to\GeoScope_Agent
.\.venv\Scripts\Activate.ps1
python -m pip install openai
```

Also add `openai` to `requirements.txt`.

### 4. Create the secrets file

Copy:

```text
.streamlit/secrets.toml.example
```

to:

```text
.streamlit/secrets.toml
```

PowerShell:

```powershell
Copy-Item .\.streamlit\secrets.toml.example .\.streamlit\secrets.toml
```

Edit the new file:

```toml
GEOSCOPE_PROVIDER = "openai"
OPENAI_API_KEY = "your-own-api-key"
OPENAI_GENERATION_MODEL = "gpt-5-mini"
OPENAI_JUDGE_MODEL = "gpt-5-mini"
```

### 5. Start GeoScope

```powershell
python -m streamlit run GeoScope.py
```

### 6. Test

1. Open Step 3 and ask an example question.
2. Confirm that an answer is generated.
3. Open Step 4 and run Generation Evaluation.

Retrieval, AOI, STAC, DuckDB, and monitoring remain unchanged.

## Temporary PowerShell alternative

```powershell
$env:GEOSCOPE_PROVIDER = "openai"
$env:OPENAI_API_KEY = "your-own-api-key"
$env:OPENAI_GENERATION_MODEL = "gpt-5-mini"
$env:OPENAI_JUDGE_MODEL = "gpt-5-mini"
python -m streamlit run GeoScope.py
```

These variables disappear when the PowerShell session closes.

## Return to Ollama

```toml
GEOSCOPE_PROVIDER = "ollama"
OLLAMA_GENERATION_MODEL = "qwen2.5:7b-instruct"
OLLAMA_JUDGE_MODEL = "llama3.1:8b"
```

## Common errors

### `OPENAI_API_KEY is not configured`

Check that the file exists at:

```text
GeoScope_Agent/.streamlit/secrets.toml
```

Then restart Streamlit.

### `ModuleNotFoundError: No module named 'openai'`

```powershell
python -m pip install openai
```

### Authentication error

Create a new key and copy it again carefully.

### Quota or billing error

Check the API platform billing page. ChatGPT billing is separate.

## Data and privacy note

When OpenAI mode is enabled, the question, retrieved context, and generation-evaluation content are sent to the OpenAI API. Do not use confidential or restricted material unless the applicable rules permit it.

The provider code uses `store=False` for Responses API calls.
