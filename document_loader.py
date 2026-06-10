import os

DATA_DIR = "data/uploaded_docs"

def load_documents():
    documents = []

    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)

        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

                # Chunking
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                documents.extend(chunks)

    return documents