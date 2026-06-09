from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def vectorize_documents(documents):
    """
    Converts list of documents into TF-IDF vectors
    """
    if not documents:
        return None, None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    vectors = vectorizer.fit_transform(documents)
    return vectorizer, vectors