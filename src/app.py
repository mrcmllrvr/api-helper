from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.chunker import Embedder, get_chroma_client, create_or_get_collection
from src.retrieval import rag_query

embedder = Embedder()
client = get_chroma_client("./chroma_db")
collection = create_or_get_collection(client, "api_docs")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/query")
async def query_api(req: QueryRequest):
    answer = rag_query(req.query, embedder, collection)
    return {"answer": answer}
