# PDF → Core Concepts (Streamlit + OpenAI)

A minimal Streamlit app to upload a scientific paper PDF, send it to OpenAI with structured output (Pydantic), and display core concepts and a summary.

## Quickstart

```bash
# 1) Create & activate a venv (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Configure the API key
cp .env.sample .env
# edit .env and set OPENAI_API_KEY=...
# optionally set OPENAI_MODEL=gpt-4.1-mini (or another model that supports structured output + file input)

# 4) Run the app
streamlit run streamlit_app.py
```

## Notes
- The app uses your `classify_pdf` pattern with `client.beta.chat.completions.parse` and a Pydantic schema.
- You can optionally analyze only the first N pages to save tokens.
- The result can be downloaded as JSON or Markdown.
- For very long or scanned PDFs, consider pre-extracting text or OCR before sending to the model.
