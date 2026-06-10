from fastapi import FastAPI, UploadFile, File
import shutil
import os
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from document_loader import load_documents
from embeddings import generate_embeddings
from vector_store import create_faiss_index
from search import search_documents
from cache import LRUCache

query_cache = LRUCache(capacity=10)


app = FastAPI()

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 5000))

UPLOAD_FOLDER = "data/uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

documents = []
faiss_index = None


@app.get("/")
def root():
    return {"message": "AI Knowledge Assistant is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    global documents, faiss_index

    documents = load_documents()
    embeddings = generate_embeddings(documents)
    faiss_index = create_faiss_index(np.array(embeddings))

    return {"message": "Document uploaded & indexed using embeddings"}

@app.get("/ask")
def ask_question(query: str):

    cached = query_cache.get(query)
    if cached:
        return {
            "results": cached,
            "cached": True
        }

    if faiss_index is None or not documents:
        return {"error": "No documents uploaded yet"}

    results = search_documents(query, faiss_index, documents)

    query_cache.put(query, results)

    return {
        "results": results,
        "cached": False
    }




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True
    )