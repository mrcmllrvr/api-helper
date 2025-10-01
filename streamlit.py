# streamlit_app.py
import os
import glob
import json
from pathlib import Path

import numpy as np
import streamlit as st
import yaml

# -----------------------------
# Page setup (hide/collapse sidebar, wide layout)
# -----------------------------
st.set_page_config(
    page_title="API Docs Chatbot (POC2)",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Minimal CSS: hide sidebar + make chat input sticky ---
st.markdown(
    """
    <style>
      /* Hide the sidebar and hamburger completely */
      [data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="baseButton-headerNoPadding"] {
        display: none !important;
      }
      /* Make the chat input sticky at the bottom of the viewport */
      [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        padding: 0.75rem 1.25rem !important;
        background: var(--background-color);
        z-index: 9999 !important;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.06);
      }
      /* Add bottom padding to the main block so content isn't hidden behind sticky input */
      .block-container {
        padding-bottom: 8rem !important;
      }
      /* Make the tabs sit a bit lower so title has breathing room */
      .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
      }
      .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 0.75rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 API Docs Chatbot (POC2)")
st.caption(
    "Ask questions about API docs in the `data/` folder (Markdown/TXT/OpenAPI YAML/JSON). "
    "Tab 2 can scan OpenAPI files for duplicate/overlapping endpoints."
)

DOC_DIR = Path("data")
DOC_DIR.mkdir(exist_ok=True)

# -----------------------------
# Model selection (OpenAI or Azure OpenAI)
# -----------------------------
USE_AZURE = bool(os.getenv("AZURE_OPENAI_ENDPOINT"))

try:
    from openai import OpenAI, AzureOpenAI
    if USE_AZURE:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")          # your GPT-4o deployment
        EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")        # your embeddings deployment
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
        EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
except Exception:
    client = None
    CHAT_MODEL = None
    EMBED_MODEL = None

# -----------------------------
# Helpers
# -----------------------------
def _ensure_models():
    if not client or not CHAT_MODEL or not EMBED_MODEL:
        raise RuntimeError(
            "Model not configured. Set OpenAI or Azure OpenAI secrets.\n\n"
            "OpenAI: OPENAI_API_KEY, OPENAI_CHAT_MODEL, OPENAI_EMBED_MODEL\n"
            "Azure:  AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,\n"
            "        AZURE_OPENAI_CHAT_DEPLOYMENT, AZURE_OPENAI_EMBED_DEPLOYMENT"
        )

def chunk_text(txt: str, max_chars: int = 1200):
    chunks, cur, count = [], [], 0
    for line in txt.splitlines():
        if count + len(line) + 1 > max_chars and cur:
            chunks.append("\n".join(cur).strip())
            cur, count = [], 0
        cur.append(line)
        count += len(line) + 1
    if cur:
        chunks.append("\n".join(cur).strip())
    return [c for c in chunks if c]

@st.cache_data(show_spinner=False)
def load_repo_docs():
    """Load .md/.txt + OpenAPI .yaml/.json under data/ into text chunks with source labels."""
    texts, sources = [], []
    for p in glob.glob(str(DOC_DIR / "**/*"), recursive=True):
        if os.path.isdir(p):
            continue
        ext = Path(p).suffix.lower()
        try:
            if ext in (".md", ".txt"):
                raw = Path(p).read_text(encoding="utf-8", errors="ignore")
                for ch in chunk_text(raw):
                    texts.append(ch)
                    sources.append(p)
            elif ext in (".yaml", ".yml", ".json"):
                raw = Path(p).read_text(encoding="utf-8", errors="ignore")
                spec = yaml.safe_load(raw) if ext in (".yaml", ".yml") else json.loads(raw)
                base = (spec.get("info") or {}).get("title") or Path(p).stem
                for path, item in (spec.get("paths") or {}).items():
                    for method, op in (item or {}).items():
                        if not isinstance(op, dict):
                            continue
                        summary = op.get("summary") or ""
                        desc = op.get("description") or ""
                        operation_id = op.get("operationId") or ""
                        text = f"{base}\n[{method.upper()}] {path}\nsummary: {summary}\noperationId: {operation_id}\n{desc}"
                        for ch in chunk_text(text, 900):
                            texts.append(ch)
                            sources.append(f"{p} {method.upper()} {path}")
        except Exception:
            continue
    return texts, sources

def embed_texts(texts):
    _ensure_models()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T

@st.cache_resource(show_spinner=False)
def build_index():
    texts, sources = load_repo_docs()
    if not texts:
        return [], [], np.zeros((0, 1))
    embs = embed_texts(texts)
    return texts, sources, embs

def answer_with_context(question: str, k: int = 5):
    texts, sources, embs = build_index()
    if len(texts) == 0:
        return "No docs found in `data/`. Add .md/.txt/.yaml/.json files first.", []

    q_emb = embed_texts([question])
    sims = cosine_sim(q_emb, embs)[0]
    idx = np.argsort(-sims)[:k]
    context_blocks = [texts[i] for i in idx]
    context_srcs = [sources[i] for i in idx]

    prompt = f"""You are an expert API Documentation Assistant for our internal API inventory.

Rules (very important):
- Answer ONLY using the “Context” provided. If the information is not present, say “I don’t know from the provided docs.”
- Always cite the specific file paths and endpoints you used (e.g., data/<file>.yml GET /v1/...).
- Be concise but complete; use short sections or bullets. Include brief code examples when useful.

Capabilities:
1) API documentation support: explain endpoints, params, auth, request/response formats, errors, rate limits.
2) API discovery: when asked “which API should I use…?” recommend the best options found in context and why.
3) Duplicate awareness: if similar endpoints appear in context, point them out and suggest which to prefer.
4) When docs are unclear/missing: say what’s missing and suggest where to look; do NOT invent details.

Tone:
- Helpful, precise, and grounded in the docs. No speculation.


Question:
{question}

Context (top {k} chunks):
{"\n\n---\n\n".join(context_blocks)}

Answer clearly, then list the sources you used."""
    _ensure_models()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    return answer, context_srcs

def load_openapi_ops():
    """Flatten OpenAPI operations across all specs under data/."""
    ops = []
    for p in glob.glob(str(DOC_DIR / "**/*"), recursive=True):
        if os.path.isdir(p):
            continue
        ext = Path(p).suffix.lower()
        if ext not in (".yaml", ".yml", ".json"):
            continue
        try:
            raw = Path(p).read_text(encoding="utf-8", errors="ignore")
            spec = yaml.safe_load(raw) if ext in (".yaml", ".yml") else json.loads(raw)
            title = (spec.get("info") or {}).get("title") or Path(p).stem
            for path, item in (spec.get("paths") or {}).items():
                for method, op in (item or {}).items():
                    if not isinstance(op, dict):
                        continue
                    ops.append({
                        "api_file": str(p),
                        "api_title": title,
                        "method": method.upper(),
                        "path": path,
                        "operationId": op.get("operationId", ""),
                        "summary": (op.get("summary") or "").strip(),
                        "desc": (op.get("description") or "").strip(),
                    })
        except Exception:
            continue
    return ops

@st.cache_resource(show_spinner=False)
def duplicate_index():
    ops = load_openapi_ops()
    if not ops:
        return ops, np.zeros((0, 1))
    strings = [f"{o['method']} {o['path']} :: {o['summary']} :: {o['desc']}" for o in ops]
    embs = embed_texts(strings)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    return ops, embs

def find_duplicates(threshold: float = 0.90, top_k: int = 3):
    ops, embs = duplicate_index()
    rows = []
    if len(ops) < 2:
        return rows
    sims = embs @ embs.T
    n = len(ops)
    for i in range(n):
        order = np.argsort(-sims[i])
        count = 0
        for j in order:
            if j == i:
                continue
            if sims[i, j] >= threshold:
                rows.append((ops[i], ops[j], float(sims[i, j])))
                count += 1
                if count >= top_k:
                    break
    return rows

# -----------------------------
# UI (two tabs)
# -----------------------------
tab1, tab2 = st.tabs(["💬 Chat with API Docs", "🧭 Duplicate Endpoint Checker"])

# ---- Tab 1: Chat UI ----
with tab1:
    texts, sources = load_repo_docs()
    if not texts:
        st.warning("No files found in `data/`. Add .md/.txt or OpenAPI .yaml/.json first.")
    else:
        st.success(f"Loaded {len(texts)} chunk(s) from {len(set(sources))} document locations.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [("assistant", "Hi! Ask me anything about the API docs in the `data/` folder.")]

    # display history
    for role, msg in st.session_state["messages"]:
        with st.chat_message(role):
            st.markdown(msg)

    # sticky chat input at screen bottom (CSS above handles stickiness)
    user_q = st.chat_input("Type your question…")
    if user_q:
        st.session_state["messages"].append(("user", user_q))
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer, srcs = answer_with_context(user_q)
                except Exception as e:
                    answer, srcs = f"⚠️ {e}", []
            st.markdown(answer)
            if srcs:
                st.caption("Sources I looked at:")
                for s in srcs:
                    st.code(s, language="text")
        st.session_state["messages"].append(("assistant", answer))

# ---- Tab 2: Duplicate endpoints ----
with tab2:
    st.write("Find similar/duplicate operations across OpenAPI files in `data/`.")
    thr = st.slider("Similarity threshold", 0.70, 0.99, 0.90, 0.01)
    k = st.slider("Max matches per endpoint", 1, 10, 3, 1)

    if st.button("Scan for duplicates"):
        with st.spinner("Scanning OpenAPI specs…"):
            try:
                rows = find_duplicates(threshold=thr, top_k=k)
            except Exception as e:
                rows = []
                st.error(f"Embedding/scan failed: {e}")

        if not rows:
            st.success("No potential duplicates found (or no OpenAPI files present).")
        else:
            for a, b, s in rows[:200]:
                st.markdown(
                    f"**{a['method']} {a['path']}**  ↔  **{b['method']} {b['path']}**  ·  similarity: `{s:.2f}`"
                )
                st.caption(f"{a['api_title']} ({a['api_file']})  ↔  {b['api_title']} ({b['api_file']})")
                if a.get("summary") or b.get("summary"):
                    st.write(f"- {a['summary'] or '(no summary)'}")
                    st.write(f"- {b['summary'] or '(no summary)'}")
                st.divider()

