import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def search_documents(query, vectorizer, document_vectors, documents, top_k=3):
    """
    Searches documents using cosine similarity
    """

    if vectorizer is None or document_vectors is None:
        return []

    # Convert query to vector
    query_vector = vectorizer.transform([query])

    # Calculate similarity
    similarities = cosine_similarity(query_vector, document_vectors)[0]

    # Get top results
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "content": documents[idx],
            "score": float(similarities[idx])
        })

    return results