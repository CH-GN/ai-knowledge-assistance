import os
import PyPDF2

def load_documents(folder_path="data/uploaded_docs"):
    documents = []

    if not os.path.exists(folder_path):
        return documents

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.lower().endswith(".pdf"):
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                documents.append(text)

        elif filename.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read())

    return documents