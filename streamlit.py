import os
import glob
from pathlib import Path
import numpy as np
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# =========================
# Page Config
# =========================
st.set_page_config(page_title="API Inventory Dashboard", layout="wide")

DOC_DIR = Path("data")
DOC_DIR.mkdir(exist_ok=True)

# =========================
# Model setup (OpenAI)
# =========================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("Missing OpenAI API key. Set OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Embeddings for search/duplicates
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# Helper functions
# =========================
def load_docs():
    docs = {}
    for p in DOC_DIR.glob("*"):
        if p.suffix.lower() in (".txt", ".md", ".yaml", ".yml", ".json"):
            try:
                docs[p.name] = p.read_text(encoding="utf-8")
            except Exception as e:
                st.warning(f"Could not read {p.name}: {e}")
    return docs

def chunk(text, size=1000):
    return [text[i:i+size] for i in range(0, len(text), size)]

@st.cache_resource
def build_doc_index():
    docs = load_docs()
    texts, meta = [], []
    for fname, content in docs.items():
        parts = chunk(content, 1000)
        texts.extend(parts)
        meta.extend([fname]*len(parts))
    if not texts:
        return None, None, None
    embs = EMBED_MODEL.encode(texts, normalize_embeddings=True)
    return texts, embs, meta

def answer_with_context(question: str):
    texts, embs, meta = build_doc_index()
    if texts is None:
        return "No documentation found.", []

    q_emb = EMBED_MODEL.encode([question], normalize_embeddings=True)
    sims = np.dot(embs, q_emb.T).squeeze()
    top_idx = np.argsort(-sims)[:5]
    context_blocks = [f"From {meta[i]}:\n{texts[i]}" for i in top_idx]

    SYSTEM_PROMPT = (
        "You are an expert API Documentation Assistant.\n"
        "Answer using ONLY the provided context. If not present, say: "
        "'I don’t know from the provided docs.'\n"
        "Be concise, structured, and cite the filenames you used."
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n\n" + '\n\n---\n\n'.join(context_blocks)},
        ],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    sources = list({meta[i] for i in top_idx})
    return answer, sources

@st.cache_resource
def duplicate_index():
    docs = load_docs()
    endpoints, base_texts = [], []
    for fname, content in docs.items():
        for line in content.splitlines():
            low = line.strip().lower()
            if low.startswith(("get ", "post ", "put ", "delete ", "patch ")):
                endpoints.append(f"{fname} | {line.strip()}")
                base_texts.append(line.strip())
    if not endpoints:
        return [], None
    embs = EMBED_MODEL.encode(base_texts, normalize_embeddings=True)
    return endpoints, embs

def find_duplicates(threshold: float = 0.90, top_k: int = 3):
    ops, embs = duplicate_index()
    rows = []
    if len(ops) < 2:
        return rows
    sims = embs @ embs.T
    n = len(ops)
    seen_pairs = set()
    for i in range(n):
        order = np.argsort(-sims[i])
        count = 0
        for j in order:
            if j == i:
                continue
            sim = float(sims[i, j])
            if sim < threshold:
                break
            key = tuple(sorted((i, j)))
            if key in seen_pairs:
                continue
            rows.append((ops[i], ops[j], sim))
            seen_pairs.add(key)
            count += 1
            if count >= top_k:
                break
    return rows

def extract_quick_endpoints(text: str, max_show=25):
    eps = []
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith(("get ", "post ", "put ", "delete ", "patch ")):
            eps.append(s)
            if len(eps) >= max_show:
                break
    return eps

# =========================
# Sidebar: Chat (auto-scroll)
# =========================
with st.sidebar:
    st.subheader("💬 Chat with API Docs")
    if "chat" not in st.session_state:
        st.session_state.chat = []

    chat_container = st.container()

    with chat_container:
        for role, msg in st.session_state.chat[-12:]:
            role_tag = "👤" if role == "user" else "🤖"
            st.markdown(f"**{role_tag} {role.capitalize()}:** {msg}")

    user_q = st.text_area("Type your question…", height=90, key="chat_input")
    send = st.button("Send", use_container_width=True)
    if send and user_q.strip():
        st.session_state.chat.append(("user", user_q.strip()))
        with st.spinner("Thinking…"):
            ans, srcs = answer_with_context(user_q.strip())
        if srcs:
            ans += "\n\n*Sources:* " + ", ".join(srcs)
        st.session_state.chat.append(("assistant", ans))
        st.experimental_rerun()

    st.markdown(
        """
        <script>
        var chatContainer = window.parent.document.querySelector('[data-testid="stSidebar"] section');
        if (chatContainer) { chatContainer.scrollTop = chatContainer.scrollHeight; }
        </script>
        """,
        unsafe_allow_html=True,
    )

# =========================
# Main: horizontally divided sections
# =========================
col_left, col_right = st.columns(2, gap="large")

# ---- Left: API Documentation Viewer ----
with col_left:
    st.header("📘 API Documentation Viewer")
    files = sorted([p for p in DOC_DIR.glob("*") if p.is_file()])
    if not files:
        st.info("No API docs found in `data/`. Add .md/.txt/.yaml/.yml/.json files.")
    else:
        for f in files:
            try:
                content = f.read_text(encoding="utf-8")
            except Exception as e:
                st.warning(f"Could not read {f.name}: {e}")
                continue
            with st.expander(f"{f.name}", expanded=False):
                eps = extract_quick_endpoints(content, max_show=50)
                if eps:
                    st.markdown("**Endpoints detected (preview):**")
                    for e in eps:
                        st.code(e, language="text")
                else:
                    st.caption("No explicit 'METHOD /path' lines detected in preview.")
                if st.checkbox(f"Show full content of {f.name}", key=f"full_{f.name}"):
                    lang = (
                        "yaml" if f.suffix.lower() in (".yaml", ".yml")
                        else "json" if f.suffix.lower() == ".json"
                        else "markdown"
                    )
                    st.code(content, language=lang)

# ---- Right: Duplicate Endpoint Checker ----
with col_right:
    st.header("🧭 Duplicate Endpoint Checker")
    st.caption("Similarity threshold is fixed at 0.90")
    top_k = st.slider("Max matches per endpoint", 1, 10, 3)
    if st.button("Scan for Duplicates", use_container_width=True):
        with st.spinner("Scanning…"):
            results = find_duplicates(threshold=0.90, top_k=top_k)
        if not results:
            st.success("✅ No potential duplicates found.")
        else:
            st.info(f"Found {len(results)} potential duplicate pairs.")
            for a, b, sim in results:
                a_file, a_op = a.split("|", 1)
                b_file, b_op = b.split("|", 1)
                with st.expander(f"{a_op.strip()} ↔ {b_op.strip()} · similarity: {sim:.2f}", expanded=False):
                    st.markdown("**Source files:**")
                    st.markdown(f"- {a_file.strip()}")
                    st.markdown(f"- {b_file.strip()}")
                    st.markdown("**Why they look similar (text compared):**")
                    st.code(a_op.strip(), language="text")
                    st.code(b_op.strip(), language="text")
