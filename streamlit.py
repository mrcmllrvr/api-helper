import os, glob, json
from pathlib import Path
import streamlit as st
import yaml
from openai import OpenAI

st.set_page_config(page_title="API Docs Chatbot (POC2)", page_icon="📚")
st.title("📚 API Docs Chatbot (POC2)")
st.caption("Ask questions about API docs in the data/ folder (Markdown/TXT/OpenAPI YAML/JSON).")

# ---- load docs from data/ ----
def load_docs():
    docs = []
    for p in glob.glob("data/**/*", recursive=True):
        if os.path.isdir(p): 
            continue
        ext = Path(p).suffix.lower()
        try:
            if ext in [".md", ".txt"]:
                docs.append((p, Path(p).read_text(encoding="utf-8", errors="ignore")))
            elif ext in [".yaml", ".yml", ".json"]:
                raw = Path(p).read_text(encoding="utf-8", errors="ignore")
                spec = yaml.safe_load(raw) if ext in (".yaml",".yml") else json.loads(raw)
                title = (spec.get("info") or {}).get("title") or Path(p).stem
                docs.append((p, f"OpenAPI spec: {title}\n{raw[:20000]}"))  # keep it simple
        except Exception:
            continue
    return docs

docs = load_docs()
if not docs:
    st.warning("No files found in `data/`. Add some .md/.txt or OpenAPI .yaml/.json files to that folder.")
else:
    st.success(f"Loaded {len(docs)} document(s).")

# ---- ask a question ----
q = st.text_area("Your question", placeholder="e.g., Which endpoint creates a user?")
if st.button("Ask"):
    if not q.strip():
        st.error("Type a question first.")
    else:
        # minimal RAG-style prompt: jam all docs as context (works for small demos)
        context = "\n\n---\n\n".join([f"[{p}]\n{txt[:4000]}" for p, txt in docs])[:120000]
        prompt = f"""You answer questions about these API docs.

Question: {q}

Docs:
{context}

Answer clearly and cite the file paths you used.
"""
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                messages=[{"role":"user","content": prompt}],
                temperature=0.2,
            )
            st.write(resp.choices[0].message.content)
        except Exception as e:
            st.error("Model call failed. Did you set OPENAI_API_KEY in Streamlit Secrets?")
