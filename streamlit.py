# streamlit_app.py
import os, glob, json
from pathlib import Path
import numpy as np
import streamlit as st
import yaml

# =========================
#  Model selection (OpenAI or Azure OpenAI)
# =========================
USE_AZURE = bool(os.getenv("AZURE_OPENAI_ENDPOINT"))

try:
    from openai import OpenAI, AzureOpenAI
    if USE_AZURE:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")          # e.g., your GPT-4o deployment name
        EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")        # e.g., your embeddings deployment name
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
        EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
except Exception:
    client = None
    CHAT_MODEL = None
    EMBED_MODEL = None

# =========================
#  Streamlit page setup
# =========================
st.set_page_config(page_title="API Docs Chatbot (POC2)", page_icon="📚", layout="wide")
st.title("📚 API Docs Chatbot (POC2)")
st.caption("Ask questions about API docs in the `data/` folder. Tab 2 can scan OpenAPI files for duplicate/overlapping endpoints.")

DOC_DIR = Path("data")
DOC_DIR.mkdir(exist_ok=True)

# =========================
#  Utilities
# =========================
def chunk_text(txt, max_chars=1200):
    chunks, cur = [], []
    count = 0
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
    """Load .md/.txt and OpenAPI .yaml/.json under data/ as chunks + sources."""
    texts, sources = [], []
    for p in glob.glob(str(DOC_DIR / "**/*"), recursive=True):
        if os.path.isdir(p):
            continue
        ext = Path(p).suffix.lower()
        try:
            if ext in [".md", ".txt"]:
                raw = Path(p).read_text(encoding="utf-8", errors="ignore")
                for ch in chunk_text(raw):
                    texts.append(ch)
                    sources.append(p)
            elif ext in [".yaml", ".yml", ".json"]:
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
    if not client or not EMBED_MODEL:
        raise RuntimeError("Model client not configured. Set your API secrets.")
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

def answer_with_context(question, k=5):
    texts, sources, embs = build_index()
    if len(texts) == 0:
        return "No docs found in data/. Add .md/.txt/.yaml/.json files.", []

    q_emb = embed_texts([question])
    sims = cosine_sim(q_emb, embs)[0]
    idx = np.argsort(-sims)[:k]
    context_blocks_
