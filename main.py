from fastapi import FastAPI, UploadFile, File
import shutil
import os

from dotenv import load_dotenv
load_dotenv()

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 5000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
print("✅ main.py LOADED")

from document_loader import load_documents
from vectorizer import vectorize_documents
from search import search_documents

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is working"}

UPLOAD_FOLDER = "data/uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

documents = []
document_vectors = None


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    global documents, document_vectors
    documents = load_documents()
    document_vectors = vectorize_documents(documents)

    return {"message": "Document uploaded & indexed successfully"}


@app.get("/ask")
def ask_question(query: str):
    if not documents:
        return {"error": "No documents uploaded yet"}

    answer, confidence = search_documents(query, document_vectors, documents)

    return {
        "answer": answer[:500],
        "confidence_score": round(confidence, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",   # change here
        port=5000,          # change here
        reload=True
    )