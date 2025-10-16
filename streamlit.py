import os
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import streamlit as st
import yaml

# =============================
# Page setup
# =============================
st.set_page_config(
    page_title="API Lens",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔎 API Lens")
st.markdown("Explore internal API docs, spot duplicate endpoints, and ask questions in the sidebar.")
st.divider()

DOC_DIR = Path("data")
DOC_DIR.mkdir(exist_ok=True)

# =============================
# OpenAI / Azure setup
# =============================
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

# =============================
# Helper functions
# =============================
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
    """Load .md/.txt + OpenAPI .yaml/.json under data/ into text chunks (for RAG)."""
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

# ---------- Duplicate checker logic ----------
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
                        "tags": op.get("tags") or [],
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

# ---------- Viewer parsing (for nested expanders) ----------
def parse_openapi_for_view(file_path: Path) -> Tuple[str, Dict[str, List[Dict]]]:
    """
    Returns (title, groups) where groups is dict[tag_or_group] -> list of ops.
    If OpenAPI tags exist, group by tag. Otherwise group by first path segment.
    """
    ext = file_path.suffix.lower()
    if ext not in (".yaml", ".yml", ".json"):
        return (file_path.name, {})
    try:
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        spec = yaml.safe_load(raw) if ext in (".yaml", ".yml") else json.loads(raw)
        title = (spec.get("info") or {}).get("title") or file_path.stem
        groups: Dict[str, List[Dict]] = defaultdict(list)

        for path, item in (spec.get("paths") or {}).items():
            for method, op in (item or {}).items():
                if not isinstance(op, dict):
                    continue
                tags = op.get("tags") or []
                summary = (op.get("summary") or "").strip()
                op_row = {"method": method.upper(), "path": path, "summary": summary}
                if tags:
                    for t in tags:
                        groups[t].append(op_row)
                else:
                    seg = path.strip("/").split("/", 1)[0] or "root"
                    groups[seg].append(op_row)

        for k in list(groups.keys()):
            groups[k].sort(key=lambda r: (r["path"], r["method"]))
        return (title, dict(sorted(groups.items(), key=lambda kv: kv[0].lower())))
    except Exception:
        return (file_path.name, {})

# =============================
# Sidebar Chat (same logic), input glued to sidebar bottom & width
# =============================
with st.sidebar:
    st.subheader("🗨️Chat with API Lens")

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
                st.markdown("**Sources I looked at:**")
                for s in srcs:
                    st.code(s, language="text")
        st.session_state["messages"].append(("assistant", answer))

# Make the chat input stick to bottom + full sidebar width
st.markdown(
    """
    <style>
      /* Position sidebar for reference */
      [data-testid="stSidebar"] {
        position: relative;
      }
      
      /* Fix chat input to bottom of sidebar only */
      [data-testid="stSidebar"] [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: auto !important;
        width: var(--sidebar-width, 21rem) !important;
        max-width: 21rem !important;
        background-color: var(--background-color);
        padding: 0.75rem 1rem;
        border-top: 1px solid rgba(250, 250, 250, 0.2);
        z-index: 999;
      }
      
      /* Make input field span full sidebar width */
      [data-testid="stSidebar"] [data-testid="stChatInput"] > div {
        width: 100% !important;
      }
      
      /* Add bottom padding to sidebar content so messages don't hide behind input */
      [data-testid="stSidebar"] > div:first-child {
        padding-bottom: 80px !important;
      }
      
      /* Ensure sidebar content is scrollable */
      [data-testid="stSidebar"] {
        overflow-y: auto;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================
# Main: side-by-side layout
# =============================
col1, col2 = st.columns([2, 3], gap="large")

# Left: API Documentation Viewer (nested expanders)
with col1:
    st.header("API Documentation Viewer")
    st.markdown("Open a file to browse its endpoints. Toggle **Show raw file** to view the full spec.")

    files = sorted([p for p in DOC_DIR.glob("*") if p.is_file()])
    if not files:
        st.info("No API docs found in `data/`.")
    else:
        for f in files:
            with st.expander(f"{f.name}", expanded=False):
                title, groups = parse_openapi_for_view(f)
                if groups:
                    st.markdown(f"**Title:** {title}")
                    for tag, ops in groups.items():
                        with st.expander(f"{tag}", expanded=False):
                            for op in ops:
                                st.markdown(f"- `{op['method']} {op['path']}` &nbsp;— {op['summary'] or '*no summary*'}")
                else:
                    st.caption("No OpenAPI structure detected. Showing raw content below.")

                show_raw = st.checkbox(f"Show raw file: {f.name}", key=f"raw_{f.name}")
                if show_raw:
                    try:
                        content = f.read_text(encoding="utf-8")
                    except Exception as e:
                        content = f"(Unable to read: {e})"
                    st.markdown(
                        f"<div style='max-height:420px; overflow-y:auto; white-space:pre-wrap; font-family:monospace;'>"
                        f"{content}</div>",
                        unsafe_allow_html=True,
                    )

# Right: Duplicate Endpoint Checker (threshold fixed internally at 0.90)
with col2:
    st.header("Duplicate Endpoint Checker")
    st.markdown("Scan all OpenAPI files in `data/` for similar or overlapping endpoints.")

    k = st.slider("Max matches per endpoint", 1, 10, 3, 1)

    if st.button("Scan for duplicates", use_container_width=True):
        with st.spinner("Scanning OpenAPI specs…"):
            try:
                pairs = find_duplicates(threshold=0.90, top_k=k)  # fixed threshold (hidden)
            except Exception as e:
                pairs = []
                st.error(f"Embedding/scan failed: {e}")

        if not pairs:
            st.success("✅ No potential duplicates found.")
        else:
            # Group by anchor endpoint (the 'a' side returned by find_duplicates)
            groups: Dict[str, Dict] = {}
            for a, b, sim in pairs:
                key = f"{a['method']} {a['path']}|{a['api_file']}"
                if key not in groups:
                    groups[key] = {
                        "anchor": a,
                        "members": []
                    }
                groups[key]["members"].append((b, sim))

            st.info(f"Found {len(groups)} groups of potential duplicate endpoints.")

            # Render each group with a side-by-side table (columns: Summary | Description | Operation ID; rows: endpoints)
            for gkey, g in groups.items():
                a = g["anchor"]
                members = sorted(g["members"], key=lambda x: -x[1])  # sort by similarity desc

                title = f"{a['method']} {a['path']}  ·  {a['api_title']} ({a['api_file']})"
                with st.expander(title, expanded=False):

                    def row_html(op, label_extra=""):
                        # Summary cell includes the endpoint label; the other two columns are description and op id
                        summary = op.get("summary") or "(no summary)"
                        desc = op.get("desc") or "(no description)"
                        opid = op.get("operationId") or "(none)"
                        endpoint = f"<b>{op['method']} {op['path']}</b>"
                        if label_extra:
                            endpoint += f" <span style='opacity:.65'>({label_extra})</span>"
                        summary_cell = f"{endpoint}<br/>{summary}"
                        return f"<tr><td>{summary_cell}</td><td>{desc}</td><td>{opid}</td></tr>"

                    # Build table: columns are Summary | Description | Operation ID
                    table_html = [
                        "<table style='width:100%; border-collapse:collapse;'>",
                        "<thead>",
                        "<tr>",
                        "<th style='text-align:left; border-bottom:1px solid #ddd;'>Summary</th>",
                        "<th style='text-align:left; border-bottom:1px solid #ddd;'>Description</th>",
                        "<th style='text-align:left; border-bottom:1px solid #ddd;'>Operation ID</th>",
                        "</tr>",
                        "</thead>",
                        "<tbody>",
                    ]

                    # Anchor row
                    table_html.append(row_html(a, "anchor"))

                    # Member rows
                    for b, sim in members:
                        label = f"match · similarity {sim:.2f} · {b['api_title']}"
                        table_html.append(row_html(b, label))

                    table_html.extend(["</tbody>", "</table>"])
                    scrollable_html = (
                        "<div style='overflow-x:auto; max-width:100%; padding-bottom:8px;'>"
                        + "\n".join(table_html)
                        + "</div>"
                    )
                    st.markdown(scrollable_html, unsafe_allow_html=True)










