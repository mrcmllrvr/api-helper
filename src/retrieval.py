import openai
from typing import List, Dict
import os

from dotenv import load_dotenv

from src.chunker import Embedder, get_chroma_client, create_or_get_collection

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def retrieve(query: str, embedder, collection, top_k: int = 5):
    q_emb = embedder.embed_texts([query])[0].tolist()
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    docs = []
    for doc, md in zip(res["documents"][0], res["metadatas"][0]):
        docs.append({"text": doc, "metadata": md})
    return docs

def build_prompt(query: str, retrieved_docs: List[Dict]) -> str:
    context_blocks = []
    for d in retrieved_docs:
        md = d["metadata"]
        snippet = f"""
Provider: {md.get('provider')}
OperationId: {md.get('operationId')}
Path: {md.get('path')}
Method: {md.get('method')}
Summary: {md.get('summary')}
Content: {d['text'][:1000] + ("..." if len(d['text']) > 1000 else "")}
"""
        context_blocks.append(snippet)

    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
You are a helpful API documentation assistant.

The user asked: {query}

Here are some relevant API docs retrieved from the database:
{context_text}

Instructions:
1. Use the retrieved information to answer the query clearly.
2. If you detect *duplicate APIs* (different providers with different names/syntax but same functionality),
   point them out explicitly. List both providers and explain the duplication.
3. Provide a concise, coherent final answer to the user.

Answer:
"""
    return prompt

def ask_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def rag_query(user_query: str, embedder, collection, top_k: int = 5, model: str = "gpt-4o-mini"):
    retrieved = retrieve(user_query, embedder, collection, top_k)
    prompt = build_prompt(user_query, retrieved)
    answer = ask_llm(prompt, model=model)
    return answer


if __name__ == "__main__":
    embedder = Embedder()
    client = get_chroma_client("./chroma_db")
    collection = create_or_get_collection(client, "api_docs")

    query = "How can I create Assistants?"
    answer = rag_query(query, embedder, collection, top_k=5, model="gpt-4o-mini")
    print("\n=== Final Answer ===\n")
    print(answer)

