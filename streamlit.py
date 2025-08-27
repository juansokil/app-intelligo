# streamlit run streamlit.py
# PDF → Intelligo Patent Search (EN-only queries, ES summary & descriptions, 3 options side-by-side)
# Muestra:
#   • Resumen del proyecto (2–3 oraciones, en español)
#   • Tres propuestas (Restrictiva / Normal / Permisiva) en cajas, cada una con:
#       - explicación en español (2–3 oraciones, sin mostrar la query)
#       - link "Abrir en Intelligo" (search= lleva el boolean TI/AB completo en inglés)

import os
import json
import base64
import unicodedata
from typing import List, Optional, Literal
from urllib.parse import quote_plus

from pydantic import BaseModel, Field, conlist
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv

# -------------------- Config --------------------
load_dotenv()
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # soporte output estructurado + file input
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_TERMS_FOR_LINK = int(os.getenv("MAX_TERMS_FOR_LINK", "3"))  # sin UI

def _secret(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

APP_USERNAME = os.getenv("APP_USERNAME") or _secret("APP_USERNAME", "intelligo")
APP_PASSWORD = os.getenv("APP_PASSWORD") or _secret("APP_PASSWORD", "r1cyt")

if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY no está seteada. Usá .env con OPENAI_API_KEY=sk-... o variable de entorno.")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------- Schema --------------------
class IPCFilter(BaseModel):
    code: str = Field(..., description="IPC like G06T or G02B27/ or A63F13/")
    level: Literal["ipc4", "ipc6"] = Field(..., description="ipc4=4-char (e.g., G06T); ipc6=group with slash (e.g., G02B27/, A63F13/)")

class YearRange(BaseModel):
    from_: int = Field(..., alias="from")
    to: int

class Theme(BaseModel):
    name: str
    keywords: List[str] = Field(default_factory=list)
    ipc_suggestions: List[str] = Field(default_factory=list)

class Refinement(BaseModel):
    name: str
    query: str
    ipc_filters: List[IPCFilter] = Field(default_factory=list)

class QueryVariant(BaseModel):
    kind: Literal["restrictive", "normal", "permissive"]
    query_string: str                   # keywords-only (EN), wrapped in parentheses (NO se muestra en UI)
    search_terms: conlist(str, min_length=1, max_length=6)   # EN tokens/phrases
    operator: Literal["OR", "AND"] = "OR"                    # para Intelligo link
    ipc_filters: List[IPCFilter] = Field(default_factory=list)
    notes: Optional[str] = None

class PatentQueryPackage(BaseModel):
    # Top-level (refleja la NORMAL) — no la mostramos en esta UI
    query_string: str
    search_terms: conlist(str, min_length=1, max_length=6)
    ipc_filters: List[IPCFilter] = Field(default_factory=list)

    query_variants: conlist(QueryVariant, min_length=3, max_length=3)

    # NUEVO: resumen del proyecto en español
    project_summary_es: Optional[str] = None

    # Extras (no se muestran acá)
    year_range: Optional[YearRange] = None
    themes: List[Theme] = Field(default_factory=list)
    refinement_queries: List[Refinement] = Field(default_factory=list)
    notes: Optional[str] = None

# -------------------- System prompt (EN-only queries + ES summary) --------------------
SYSTEM_MESSAGE = """\
You are a patent landscape assistant. Return STRICT JSON matching the PatentQueryPackage schema. No extra keys or text.

LANGUAGE REQUIREMENTS:
- All query strings and search terms MUST be in ENGLISH (for patent retrieval).
- Additionally, return project_summary_es as 2–3 sentences in SPANISH summarizing the project in plain language.

Produce THREE query variants after analyzing the document:
- restrictive: high precision, narrower scope (can use AND; fewer but specific terms; targeted IPC filters).
- normal: optimal balance (prefer OR; well-chosen terms; modest IPC filters).
- permissive: high recall (OR; broader/general terms; minimal IPC constraints).

Rules:
- Each variant's query_string: keywords-only (ENGLISH), wrapped in parentheses; join with OR or AND; no field prefixes (no ti:/ab:), no IPC inside query_string.
- Each variant: 1–6 ENGLISH search_terms (tokens/phrases) to feed Intelligo’s `search=` Boolean builder, and operator = "OR" or "AND".
- IPC filters OPTIONAL per variant; mark as 'ipc4' (e.g., G06T) or 'ipc6' (e.g., G02B27/, A63F13/). Include only supported codes.
- Top-level query_string/search_terms/ipc_filters MUST mirror the NORMAL (optimal) variant.

Constraints:
- Use quotes for multi-word phrases (e.g., "virtual reality").
- Prefer concise, retrieval-friendly ENGLISH terms; avoid fluff.
- Return strictly valid JSON per schema.
"""

# -------------------- Helpers --------------------
def _ascii_sanitize(s: str) -> str:
    if not isinstance(s, str):
        return s
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def _normalize_terms(terms: List[str], max_terms: int) -> List[str]:
    """EN tokens 1–3 palabras, dedupe, límite, sin comillas/diacríticos."""
    cleaned = []
    for t in terms or []:
        t = _ascii_sanitize(" ".join((t or "").replace('"', "").replace("’", "'").split())).strip()
        if not t:
            continue
        words = t.split()
        if len(words) > 3:
            t = " ".join(words[:3])
        cleaned.append(t)
    uniq, seen = [], set()
    for t in cleaned:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(t)
    return uniq[:max_terms]

def _normalize_operator(op: Optional[str]) -> str:
    return "AND" if (op or "").upper() == "AND" else "OR"

def _normalize_ipc_filters(filters: List[dict]) -> List[dict]:
    """Normaliza IPC: uppercase, sin espacios, infiere nivel, evita duplicados."""
    out, seen = [], set()
    for f in filters or []:
        if isinstance(f, str):
            code_raw = f; level = None
        else:
            code_raw = f.get("code", ""); level = f.get("level")
        code = (code_raw or "").upper().replace(" ", "").strip()
        if not code:
            continue
        if not level:
            level = "ipc6" if "/" in code else ("ipc4" if len(code) == 4 else "ipc4")
        if level == "ipc4" and len(code) != 4:
            code = code[:4]
        if level == "ipc6" and "/" not in code:
            if len(code) > 4:
                if not code.endswith("/"):
                    code = code + "/"
            else:
                level = "ipc4"
        key = (code, level)
        if key in seen:
            continue
        seen.add(key)
        out.append({"code": code, "level": level})
    return out

def build_intelligo_search_string(terms: List[str], operator: str = "OR") -> str:
    """Boolean para Intelligo `search=`: (ti:term1 OR ab:term1) OP (ti:term2 OR ab:term2) ..."""
    cleaned = []
    for t in terms or []:
        if not t:
            continue
        t = _ascii_sanitize(" ".join(t.replace('"', "").replace("’", "'").split())).strip()
        if not t:
            continue
        cleaned.append(f"(ti:{t} OR ab:{t})")
    if not cleaned:
        return ""
    return f" {operator} ".join(cleaned)

def build_intelligo_url_from_variant(terms: List[str], ipc_filters: List[dict], operator: str = "OR") -> str:
    """
    Intelligo: `search=` con boolean TI/AB (URL-encoded) + `filtersArray=ipc4:...`/`ipc6:...` repetidos.
    """
    base = "https://www.explora-intelligo.info/patentes?search="
    search_str = build_intelligo_search_string(terms, operator=operator)
    url = base + quote_plus(search_str)
    for f in ipc_filters or []:
        level = (f.get("level") or "ipc4").strip()
        code = (f.get("code") or "").strip()
        if not code:
            continue
        url += "&filtersArray=" + quote_plus(f"{level}:{code}")
    return url

def _list_preview(items, limit=3) -> str:
    items = [i for i in (items or []) if i]
    if not items:
        return ""
    if len(items) <= limit:
        return ", ".join(items)
    return f"{', '.join(items[:limit])} +{len(items) - limit} más"

def _variant_label_es(kind: str) -> str:
    return {
        "restrictive": "🎯 Restrictiva",
        "normal": "⚖️ Normal",
        "permissive": "🌐 Permisiva",
    }.get((kind or "").lower(), (kind or "Variante").title())

def _variant_description_es(v: dict) -> str:
    """2–3 oraciones en español (no muestra la query)."""
    kind = (v.get("kind") or "").lower()
    terms = v.get("search_terms", []) or []
    op = (v.get("operator") or "OR").upper()
    ipcs = v.get("ipc_filters", []) or []
    ipc_txt = ", ".join([f"{f['level']}:{f['code']}" for f in ipcs]) if ipcs else "ninguno"

    # Cobertura
    if kind == "restrictive" or op == "AND" or len(ipcs) >= 2:
        coverage = "un foco muy preciso"
    elif kind == "permissive" or (op == "OR" and not ipcs and len(terms) >= 3):
        coverage = "una cobertura amplia (alto recall)"
    else:
        coverage = "una cobertura equilibrada"

    s1 = f"Usa {len(terms)} término{'s' if len(terms)!=1 else ''} conciso{'s' if len(terms)!=1 else ''} ({_list_preview(terms)}) combinados con {op} para {coverage}."
    s2 = f"Aplica {len(ipcs)} filtro{'s' if len(ipcs)!=1 else ''} IPC ({ipc_txt}) para orientar los resultados." if ipcs \
         else "No aplica filtros IPC, por lo que pueden aparecer dominios adyacentes."
    if kind == "restrictive":
        s3 = "Elegila cuando necesites un conjunto semilla pequeño y muy relevante para revisión manual."
    elif kind == "permissive":
        s3 = "Úsala para explorar el espacio y no perder variantes de redacción; esperá más ruido."
    else:
        s3 = "Buena opción por defecto: balancea cobertura y precisión."
    return " ".join([s1, s2, s3])

# -------------------- OpenAI call --------------------
def extract_query_from_pdf(file_path: str, base64_str: str, max_terms_for_link: int = 3):
    """
    Devuelve un dict tipo PatentQueryPackage y normaliza cada variante.
    Incluye 'project_summary_es' (2–3 oraciones).
    """
    try:
        resp = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            response_format=PatentQueryPackage,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Read the attached document and return a JSON PatentQueryPackage with three query_variants "
                                "(restrictive, normal, permissive). Each variant must include query_string (ENGLISH), "
                                "search_terms (ENGLISH), operator, and optional ipc_filters. "
                                "Also provide project_summary_es: 2–3 sentences in SPANISH summarizing the project. "
                                "Top-level fields must mirror the NORMAL variant."
                            ),
                        },
                        {
                            "type": "file",
                            "file": {
                                "filename": file_path,
                                "file_data": f"data:application/pdf;base64,{base64_str}",
                            },
                        },
                    ],
                },
            ],
        )
        parsed = resp.choices[0].message.parsed
        data = json.loads(parsed.model_dump_json())

        # Normalización de variantes
        variants = data.get("query_variants", []) or []
        for v in variants:
            v["query_string"] = _ascii_sanitize(v.get("query_string", "") or "")
            v["search_terms"] = _normalize_terms(v.get("search_terms", []), max_terms=max(1, min(6, max_terms_for_link)))
            v["operator"] = _normalize_operator(v.get("operator"))
            v["ipc_filters"] = _normalize_ipc_filters(v.get("ipc_filters", []))
        data["query_variants"] = variants

        # project_summary_es (dejamos tildes y caracteres tal cual)
        if not data.get("project_summary_es"):
            data["project_summary_es"] = None

        return data
    except Exception as e:
        return {"error": str(e)}

# -------------------- Page + Auth --------------------
st.set_page_config(page_title="PDF → Intelligo (3 propuestas)", page_icon="🔎", layout="wide")

def require_login():
    if st.session_state.get("auth", False):
        st.sidebar.success(f"Sesión: {st.session_state.get('user', 'usuario')}")
        if st.sidebar.button("Cerrar sesión"):
            st.session_state.clear()
            st.rerun()
        return True

    with st.sidebar.form("login_form"):
        st.subheader("Ingresar")
        user = st.text_input("Usuario")
        pwd = st.text_input("Contraseña", type="password")
        submit = st.form_submit_button("Entrar")

    if submit:
        if user == APP_USERNAME and pwd == APP_PASSWORD:
            st.session_state["auth"] = True
            st.session_state["user"] = user
            st.success("Sesión iniciada.")
            st.rerun()
        else:
            st.error("Credenciales inválidas.")

    st.stop()

require_login()

# -------------------- UI --------------------
st.title("📄→🔎 Intelligo Patent Search — Tres Propuestas")
st.caption("Subí un PDF. Vas a ver un resumen del proyecto en español y tres propuestas (Restrictiva / Normal / Permisiva) con explicación en español y link directo a Intelligo. Las búsquedas/keywords se generan en inglés (para patentes).")

uploaded = st.file_uploader("Subí un PDF técnico/proyecto", type=["pdf"])

if uploaded is not None:
    st.info(f"Archivo seleccionado: **{uploaded.name}** ({uploaded.size/1024:.1f} KB)")
    if st.button("Generar propuestas"):
        with st.spinner("Analizando el PDF y generando resumen y propuestas…"):
            uploaded.seek(0)
            b64 = base64.b64encode(uploaded.read()).decode("utf-8")
            result = extract_query_from_pdf(uploaded.name, b64, max_terms_for_link=MAX_TERMS_FOR_LINK)

        if "error" in result:
            st.error("Error de API: " + result["error"])
        else:
            # ----- Resumen del proyecto (ES) -----
            summary = (result.get("project_summary_es") or "").strip()
            if summary:
                st.markdown("## Resumen del proyecto")
                st.write(summary)
                st.markdown("---")

            # ----- Propuestas -----
            variants = result.get("query_variants", []) or []
            if not variants:
                st.warning("No se devolvieron variantes de búsqueda.")
            else:
                order = {"restrictive": 0, "normal": 1, "permissive": 2}
                variants = sorted(variants, key=lambda v: order.get((v.get("kind") or "").lower(), 1))

                st.markdown("## Propuestas")
                cols = st.columns(3)
                for idx, v in enumerate(variants[:3]):
                    with cols[idx]:
                        box = st.container(border=True)
                        with box:
                            st.markdown(f"#### {_variant_label_es(v.get('kind'))}")
                            # SOLO explicación en español (no mostramos la query)
                            st.caption(_variant_description_es(v))
                            # Link a Intelligo
                            link = build_intelligo_url_from_variant(
                                v.get("search_terms", []) or [],
                                v.get("ipc_filters", []) or [],
                                operator=(v.get("operator") or "OR")
                            )
                            st.markdown(f"[Abrir en Intelligo]({link})")
else:
    st.info("Subí un PDF para comenzar.")

