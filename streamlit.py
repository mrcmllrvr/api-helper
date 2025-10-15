import os
import glob
import json
from pathlib import Path
import numpy as np
import streamlit as st
import yaml
from collections import defaultdict

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="API Docs Chatbot (POC2)",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📚 API Docs Chatbot (POC2)")
st.caption(
    "Ask questions about API docs in the `data/` folder. "
    "Use the left section to view API documentation and run duplicate checks."
)

DOC_DIR = Path("data")
DOC_DIR.mkdir(exist_ok=True)

# -----------------------------
# OpenAI / Azure setup
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
        CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
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
        raise RuntimeError("⚠️ Missing model configuration for OpenAI or Azure OpenAI.")

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
    """Load .md/.txt + OpenAPI .yaml/.json under data/ into text chunks."""
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
        return "No docs found in `data/`.", []
    q_emb = embed_texts([question])
    sims = cosine_sim(q_emb, embs)[0]
    idx = np.argsort(-sims)[:k]
    context_blocks = [texts[i] for i in idx]
    context_srcs = [sources[i] for i in idx]
    prompt = f"""You are an expert API Documentation Assistant.

Answer based only on the provided context below. If unknown, say: "I don’t know from the provided docs."

Question: {question}

Context:
{"\n\n---\n\n".join(context_blocks)}
"""
    _ensure_models()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    return answer, context_srcs

# -----------------------------
# Duplicate checker logic (unchanged)
# -----------------------------
def load_openapi_ops():
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
    seen_pairs = set()
    per_endpoint_count = defaultdict(int)
    for i in range(n):
        order = np.argsort(-sims[i])
        for j in order:
            if j == i:
                continue
            sim = float(sims[i, j])
            if sim < threshold:
                break
            pair_key = (i, j) if i < j else (j, i)
            if pair_key in seen_pairs:
                continue
            if per_endpoint_count[i] >= top_k:
                break
            rows.append((ops[i], ops[j], sim))
            seen_pairs.add(pair_key)
            per_endpoint_count[i] += 1
    return rows

# -----------------------------
# Sidebar: Chat (unchanged logic)
# -----------------------------
with st.sidebar:
    st.subheader("💬 Chat with API Docs")
    texts, sources = load_repo_docs()
    if not texts:
        st.warning("No docs found in `data/`.")
    else:
        st.success(f"Loaded {len(texts)} chunks from {len(set(sources))} docs.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [("assistant", "Hi! Ask me anything about the API docs.")]

    for role, msg in st.session_state["messages"]:
        with st.chat_message(role):
            st.markdown(msg)

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

# -----------------------------
# Main UI
# -----------------------------
st.header("📘 API Documentation Viewer")
files = sorted([p for p in DOC_DIR.glob("*") if p.is_file()])

if not files:
    st.info("No API docs found in `data/`.")
else:
    for f in files:
        try:
            content = f.read_text(encoding="utf-8")
        except Exception as e:
            st.warning(f"Could not read {f.name}: {e}")
            continue

        with st.expander(f"{f.name}", expanded=False):
            # Search bar + highlight
            search_term = st.text_input(f"🔍 Search in {f.name}", "", key=f"search_{f.name}")
            highlighted_content = content

            if search_term.strip():
                # simple case-insensitive highlight
                highlighted_content = highlighted_content.replace(
                    search_term,
                    f"🟨**{search_term}**🟨"
                )
                highlighted_content = highlighted_content.replace(
                    search_term.lower(),
                    f"🟨**{search_term.lower()}**🟨"
                )
                highlighted_content = highlighted_content.replace(
                    search_term.upper(),
                    f"🟨**{search_term.upper()}**🟨"
                )

            st.markdown(
                f"<div style='max-height:420px; overflow-y:auto; white-space:pre-wrap; font-family:monospace;'>"
                f"{highlighted_content}</div>",
                unsafe_allow_html=True,
            )

st.divider()

# -----------------------------
# Duplicate Endpoint Checker
# -----------------------------
st.header("🧭 Duplicate Endpoint Checker")
st.write("Find similar or overlapping endpoints across OpenAPI specs.")

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
        st.success("✅ No potential duplicates found.")
    else:
        for a, b, s in rows[:200]:
            st.markdown(
                f"**{a['method']} {a['path']}**  ↔  **{b['method']} {b['path']}**  ·  similarity: `{s:.2f}`"
            )
            st.caption(f"{a['api_title']} ({a['api_file']})  ↔  {b['api_title']} ({b['api_file']})")
            if a.get('summary') or b.get('summary'):
                st.write(f"- {a['summary'] or '(no summary)'}")
                st.write(f"- {b['summary'] or '(no summary)'}")
            st.divider()
