#streamlit run streamlit.py

# streamlit.py
# Minimal Streamlit app: upload a PDF and get
# (1) a 2–3 sentence summary, and
# (2) N keyword-like core concepts (each 1–3 words, optimized for WIPO/Intelligo),
# plus a one-click Intelligo search URL and OPTIONAL embeddings saved into the JSON.
#
# Requires:
#   streamlit>=1.36
#   openai>=1.40.0
#   pydantic>=2.7
#   python-dotenv>=1.0
#
# Run:
#   streamlit run streamlit.py

import os
import json
import base64
from typing import List
from urllib.parse import quote_plus

from pydantic import BaseModel, Field, conlist
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv


# -------------------- Config --------------------
load_dotenv()
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # must support structured output + file input
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---- Auth settings (read from env/secrets; fallback to provided creds) ----
def _secret(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

APP_USERNAME = os.getenv("APP_USERNAME") or _secret("APP_USERNAME", "intelligo")
APP_PASSWORD = os.getenv("APP_PASSWORD") or _secret("APP_PASSWORD", "r1cyt")

if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not set. Create a .env with OPENAI_API_KEY=sk-... or set it in your environment.")

client = OpenAI(api_key=OPENAI_API_KEY)


# -------------------- Minimal Schema (tailored for Intelligo/WIPO) --------------------
class PaperMini(BaseModel):
    # 2–3 sentences, concise, neutral
    short_summary: str = Field(..., description="2–3 sentence summary (≤80 words)")
    # Up to 10 core concepts, each 1–3 words, noun-phrase style (keywords for patent search)
    core_concepts: conlist(str, min_length=1, max_length=10) = Field(
        default_factory=list,
        description="Keyword-like concepts; each is 1–3 words (noun phrases) suitable for patent search"
    )


# -------------------- System prompt --------------------
SYSTEM_MESSAGE = """\
You are an expert scientific-paper analyst creating outputs for patent prior-art searches (e.g., WIPO/Intelligo).
Return JSON that fits the PaperMini schema exactly. Do not include any extra keys or text.

Rules for short_summary:
- Write 2–3 sentences, ≤80 words total, plain English, neutral tone.
- Summarize the main idea and what is novel/useful.

Rules for core_concepts:
- Return EXACTLY the requested number of concepts unless fewer are truly warranted.
- Each concept MUST be a compact noun phrase of 1–3 words (e.g., "domain adaptation", "graph transformers", "contrastive learning").
- Prefer technical, patent-searchable terms; avoid vague words like "method", "system", "novel", "approach".
- Only include concepts clearly supported by the paper; do not invent facts.
"""


# -------------------- Helpers --------------------
def _normalize_concepts(concepts: List[str], n_concepts: int) -> List[str]:
    """Ensure each concept is 1–3 words, deduplicate, and cap at n_concepts."""
    cleaned = []
    for t in concepts or []:
        t = " ".join((t or "").replace('"', "").split()).strip()
        if not t:
            continue
        words = t.split()
        if len(words) > 3:
            t = " ".join(words[:3])  # trim to first 3 words
        cleaned.append(t)

    # de-duplicate case-insensitively
    seen = set()
    uniq = []
    for t in cleaned:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)

    return uniq[:n_concepts]


def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    """Return embeddings for the given texts (list of strings)."""
    if not texts:
        return []
    resp = client.embeddings.create(model=model, input=texts)
    # Ensure ordering matches inputs; OpenAI returns in the same order
    return [d.embedding for d in resp.data]


# -------------------- OpenAI call using structured output --------------------
def classify_pdf(file_path: str, base64_str: str, n_concepts: int = 2):
    """
    Analyze the PDF and return:
      - short_summary (2–3 sentences)
      - core_concepts: EXACTLY N compact terms (1–3 words each), suitable for patent search.
    Uses client.beta.chat.completions.parse with a Pydantic model for structured output.
    """
    try:
        resp = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            response_format=PaperMini,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze the attached scientific paper and return JSON for PaperMini. "
                                f"Return at most {n_concepts} core concepts; prefer returning {n_concepts} "
                                "unless only one is truly central. Ensure each concept is 1–3 words."
                            ),
                        },
                        {
                            "type": "file",
                            "file": {
                                "filename": file_path,
                                "file_data": f"data:application/pdf;base64,{base64_str}"
                            }
                        },
                    ],
                },
            ],
        )
        parsed = resp.choices[0].message.parsed  # Pydantic object
        data = json.loads(parsed.model_dump_json())
        data["core_concepts"] = _normalize_concepts(data.get("core_concepts", []), n_concepts)
        return data
    except Exception as e:
        return {"error": str(e)}


# -------------------- Intelligo URL builders --------------------
def build_intelligo_url_from_terms(terms: List[str], section: str = "patentes") -> str:
    """
    Build a direct Intelligo URL from multiple terms.
    We quote EACH term for phrase matching and join them with spaces.
    Example query string:  "graph transformers" "domain adaptation"
    """
    base = f"https://www.explora-intelligo.info/{section}?search="
    cleaned = []
    for t in terms or []:
        t = (t or "").replace('"', "").strip()
        if not t:
            continue
        t = " ".join(t.split())
        cleaned.append(t)
    if not cleaned:
        return base
    q = " ".join([f'"{t}"' for t in cleaned])
    return base + quote_plus(q)


def build_intelligo_url_single(term: str, section: str = "patentes") -> str:
    term = (term or "").replace('"', "").strip()
    base = f"https://www.explora-intelligo.info/{section}?search="
    if not term:
        return base
    return base + quote_plus(f'"{term}"')


# -------------------- Page + Auth --------------------
st.set_page_config(page_title="PDF → Mini Summary & Patent-Style Concepts", page_icon="📄")

def require_login():
    # Already authed?
    if st.session_state.get("auth", False):
        st.sidebar.success(f"Logged in as {st.session_state.get('user', 'user')}")
        if st.sidebar.button("Log out"):
            st.session_state.clear()
            st.rerun()
        return True

    with st.sidebar.form("login_form"):
        st.subheader("Sign in")
        user = st.text_input("User")
        pwd = st.text_input("Password", type="password")
        submit = st.form_submit_button("Log in")

    if submit:
        if user == APP_USERNAME and pwd == APP_PASSWORD:
            st.session_state["auth"] = True
            st.session_state["user"] = user
            st.success("Signed in.")
            st.rerun()
        else:
            st.error("Invalid credentials. Try again.")

    st.stop()  # Stop the script until login succeeds

require_login()

# --------------- App content (only rendered when authed) ---------------
st.title("📄→🔎 Mini Summary & Patent-Style Concepts (+ Embeddings)")
st.caption("Upload a PDF to get a 2–3 sentence summary and 2–10 compact, keyword-like concepts suitable for WIPO/Intelligo search. Optionally add embeddings to the JSON.")

with st.expander("⚙️ Settings", expanded=False):
    st.write("Model:", MODEL_NAME)
    EMBED_MODEL = st.text_input("Embeddings model", value=EMBED_MODEL, help="e.g., text-embedding-3-small / text-embedding-3-large")

uploaded = st.file_uploader("Upload a scientific paper (PDF)", type=["pdf"])
# Slider for 2–10 concepts
n_concepts = st.slider("Number of core concepts", min_value=2, max_value=10, value=2)
compute_embeddings = st.checkbox("Compute embeddings and include in JSON", value=True)
show_json = st.checkbox("Show raw JSON", value=False)

if uploaded is not None:
    st.info(f"Selected file: **{uploaded.name}** ({uploaded.size/1024:.1f} KB)")

    if st.button("Analyze"):
        with st.spinner("Calling GPT to analyze your PDF…"):
            b64 = base64.b64encode(uploaded.read()).decode("utf-8")
            result = classify_pdf(uploaded.name, b64, n_concepts=n_concepts)

        if "error" in result:
            st.error("API error: " + result["error"])
        else:
            # --- Summary ---
            st.markdown("### Summary")
            st.write(result.get("short_summary", "—"))

            # --- Core Concepts (2–10 items, each 1–3 words) ---
            concepts: List[str] = result.get("core_concepts", []) or []
            st.markdown(f"### Core Concepts ({n_concepts} terms, 1–3 words each)")
            if concepts:
                # Soft hint if any concept got trimmed to 3 words (likely longer originally)
                too_long = [c for c in concepts if len(c.split()) >= 3]
                if too_long:
                    st.caption("⚠️ Some concepts were trimmed to 3 words to meet the 1–3 word rule.")
                for i, term in enumerate(concepts, 1):
                    st.markdown(f"- **{term}**")
            else:
                st.write("No core concepts extracted.")

            # --- Intelligo links ---
            st.markdown("### Search in Intelligo (Patents)")
            if concepts:
                combined_url = build_intelligo_url_from_terms(concepts, section="patentes")
                st.markdown(f"**Combined query:** [{combined_url}]({combined_url})")
                st.code(combined_url, language="text")

                st.markdown("**Individual queries:**")
                for term in concepts:
                    url = build_intelligo_url_single(term, section="patentes")
                    st.markdown(f"- {term}: [{url}]({url})")
            else:
                st.info("Concepts are required to build Intelligo links.")

            # --- Embeddings (optional) ---
            if compute_embeddings:
                try:
                    with st.spinner("Computing embeddings…"):
                        texts = [result.get("short_summary", "")]
                        concept_texts = concepts
                        # First element is summary, then concepts (N terms)
                        vectors = embed_texts(texts + concept_texts, model=EMBED_MODEL)

                    summary_vec = vectors[0] if vectors else []
                    concept_vecs = vectors[1:] if len(vectors) > 1 else []

                    # Attach to result JSON
                    result["embeddings"] = {
                        "model": EMBED_MODEL,
                        "summary": {
                            "dim": len(summary_vec),
                            "vector": summary_vec
                        },
                        "core_concepts": [
                            {"term": term, "dim": len(vec), "vector": vec}
                            for term, vec in zip(concepts, concept_vecs)
                        ]
                    }
                    st.success(f"Embeddings added ({EMBED_MODEL}).")
                except Exception as e:
                    st.error(f"Embedding error: {e}")

            # --- Download JSON (includes embeddings if computed) ---
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(result, indent=2).encode("utf-8"),
                file_name="paper_mini_summary.json",
                mime="application/json",
            )

            if show_json:
                st.markdown("### Raw JSON")
                st.code(json.dumps(result, indent=2))

else:
    st.info("Upload a PDF to begin.")