import faiss
import numpy as np

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_index(index, query_embedding, top_k=3):
    distances, indices = index.search(query_embedding, top_k)
    return indices[0], distances[0]