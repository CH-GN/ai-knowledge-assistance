import numpy as np
from embeddings import generate_embeddings
from vector_store import search_index

def search_documents(query, faiss_index, documents, top_k=3):
    """
    Semantic search using sentence embeddings + FAISS
    """

    if faiss_index is None or not documents:
        return []

    # Convert query → embedding
    query_embedding = generate_embeddings([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS index
    indices, distances = search_index(faiss_index, query_embedding, top_k)

    results = []
    for idx, dist in zip(indices, distances):
        score = float(1 / (1 + dist))  # convert distance → confidence
        results.append({
            "content": documents[idx],
            "score": round(score, 3)
        })

    return results
    
from ranking import get_top_k

def search_documents(query, index, documents, k=3):
    query_embedding = generate_embeddings([query])
    indices, distances = search_index(index, query_embedding, top_k=10)

    results = []
    scores = []

    for i, dist in zip(indices, distances):
        results.append(documents[i])
        scores.append(float(1 / (1 + dist)))

    top_results = get_top_k(results, scores, k)

    return top_results