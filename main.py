from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import numpy as np
from dotenv import load_dotenv

from document_loader import load_documents
from embeddings import generate_embeddings
from vector_store import create_faiss_index
from search import search_documents
from cache import LRUCache
from answer_builder import build_human_answer

# ------------------ INIT ------------------
load_dotenv()
app = FastAPI(title="AI Knowledge Assistant")

# ------------------ CONFIG ------------------
UPLOAD_FOLDER = "data/uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

query_cache = LRUCache(capacity=10)

documents = []
faiss_index = None

# ------------------ HELPERS ------------------
def is_summary_question(query: str) -> bool:
    summary_keywords = [
        "summary",
        "conclusion",
        "overall",
        "overview",
        "brief",
        "in short",
        "final"
    ]
    query = query.lower()
    return any(word in query for word in summary_keywords)

# ------------------ ROUTES ------------------
@app.get("/")
def root():
    return {"message": "AI Knowledge Assistant is running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        global documents, faiss_index

        documents = load_documents()
        if documents is None or len(documents) == 0:
            raise HTTPException(400, "No documents loaded")

        embeddings = generate_embeddings(documents)
        if embeddings is None or len(embeddings) == 0:
            raise HTTPException(500, "Embedding generation failed")

        embeddings = np.array(embeddings, dtype="float32")

        if embeddings.shape[0] == 0:
            raise HTTPException(500, "Empty embeddings array")

        faiss_index = create_faiss_index(embeddings)

        return {"message": "Document uploaded & indexed successfully"}

    except Exception as e:
        print("UPLOAD ERROR:", e)
        raise HTTPException(500, str(e))

@app.get("/ask")
def ask_question(query: str):
    global documents, faiss_index

    cached = query_cache.get(query)
    if cached:
        return {"answer": cached, "cached": True}

    if not documents or faiss_index is None:
        raise HTTPException(400, "Upload documents first")

    results = search_documents(query, faiss_index, documents)
    if not results:
        return {"answer": "No relevant information found."}

    if is_summary_question(query):
        human_answer = build_human_answer(results)
    else:
        human_answer = results[0][0]  # text only

    query_cache.put(query, human_answer)
    return {"answer": human_answer, "cached": False}