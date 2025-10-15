import os
import glob
from pathlib import Path
import streamlit as st
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ---------------------------
# Configuration
# ---------------------------
st.set_page_config(page_title="API Helper", layout="wide")

# ---- Sidebar: List available API docs ----
st.sidebar.title("📂 API Documentation Files")
DOC_DIR = Path("data")
api_files = sorted([p for p in glob.glob(str(DOC_DIR / "*")) if os.path.isfile(p)])
if not api_files:
    st.sidebar.info("No API docs found in `data/` folder.")
else:
    for file in api_files:
        st.sidebar.markdown(f"- [{Path(file).name}]({file})")

# ---------------------------
# Setup
# ---------------------------
st.title("🤖 API Helper")
st.caption("Chatbot & Duplicate Endpoint Checker powered by GPT-4o")

openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not openai_api_key:
    st.error("Missing OpenAI API key. Please set it in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=openai_api_key)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
CHAT_MODEL = "gpt-4o-mini"

# ---------------------------
# Helper functions
# ---------------------------
def load_docs():
    docs = {}
    for path in DOC_DIR.glob("*"):
        if path.suffix.lower() in [".txt", ".md", ".yaml", ".yml", ".json"]:
            try:
                docs[path.name] = path.read_text(encoding="utf-8")
            except Exception as e:
                st.warning(f"Could not read {path.name}: {e}")
    return docs

@st.cache_resource
def build_doc_index():
    docs = load_docs()
    texts, meta = [], []
    for fname, content in docs.items():
        chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
        texts.extend(chunks)
        meta.extend([fname]*len(chunks))
    if not texts:
        return None, None, None
    embs = embed_model.encode(texts, normalize_embeddings=True)
    return texts, embs, meta

@st.cache_resource
def duplicate_index():
    docs = load_docs()
    endpoints, texts = [], []
    for fname, content in docs.items():
        for line in content.splitlines():
            if line.strip().lower().startswith(("get ", "post ", "put ", "delete ")):
                endpoints.append(f"{fname} | {line.strip()}")
                texts.append(line.strip())
    if not endpoints:
        return [], None
    embs = embed_model.encode(texts, normalize_embeddings=True)
    return endpoints, embs

def answer_with_context(question: str):
    texts, embs, meta = build_doc_index()
    if texts is None:
        return "No documentation found.", []

    q_emb = embed_model.encode([question], normalize_embeddings=True)
    sims = np.dot(embs, q_emb.T).squeeze()
    top_idx = np.argsort(-sims)[:5]
    context_blocks = [f"From {meta[i]}:\n{texts[i]}" for i in top_idx]

    SYSTEM_PROMPT = """You are an expert API Documentation Assistant.
Answer questions using the provided API docs context only.
Be concise, structured, and cite the filenames you used.
If the answer isn't in context, say 'I don’t know from the provided docs.'"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n\n" + "\n\n---\n\n".join(context_blocks)},
        ],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content
    sources = list({meta[i] for i in top_idx})
    return answer, sources

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
            pair_key = tuple(sorted((i, j)))
            if pair_key in seen_pairs:
                continue
            rows.append((ops[i], ops[j], sim))
            seen_pairs.add(pair_key)
            count += 1
            if count >= top_k:
                break
    return rows

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2 = st.tabs(["💬 Chat with API Docs", "🧭 Duplicate Endpoint Checker"])

# ---------------------------
# Tab 1: Chatbot
# ---------------------------
with tab1:
    st.subheader("💬 Chat about API Documentation")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for role, msg in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(msg)

    user_q = st.chat_input("Ask something about the APIs…")
    if user_q:
        st.session_state.messages.append(("user", user_q))
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer, srcs = answer_with_context(user_q)
            st.markdown(answer)
            if srcs:
                st.caption("📚 Sources:")
                for s in srcs:
                    st.code(s, language="text")
        st.session_state.messages.append(("assistant", answer))

# ---------------------------
# Tab 2: Duplicate Checker
# ---------------------------
with tab2:
    st.subheader("🧭 Duplicate Endpoint Checker")

    # Fixed threshold at 0.90
    thr = 0.90
    st.caption("🔒 Similarity threshold is fixed at 0.90")

    top_k = st.slider("Max matches per endpoint", 1, 10, 3)
    if st.button("Scan for Duplicates"):
        with st.spinner("Scanning..."):
            results = find_duplicates(threshold=thr, top_k=top_k)
        if not results:
            st.success("✅ No potential duplicates found.")
        else:
            st.info(f"Found {len(results)} potential duplicate pairs.")
            for a, b, sim in results:
                st.markdown(f"**{a.split('|')[1].strip()} ↔ {b.split('|')[1].strip()} · similarity:** `{sim:.2f}`")
                st.markdown(
                    f"*{a.split('|')[0]}* ↔ *{b.split('|')[0]}*",
                    help="These are the API docs containing the endpoints.",
                )
                st.markdown("---")

# ---------------------------
# CSS: sticky chat input
# ---------------------------
st.markdown(
    """
    <style>
    [data-testid="stChatInput"] { position: fixed; bottom: 0; width: 80%; background: white; }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)
