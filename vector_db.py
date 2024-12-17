import numpy as np
from langchain.vectorstores import FAISS

class VectorDatabase:
    def __init__(self):
        self.embeddings = []
        self.metadata = []
        self.vector_store = FAISS()
        self.cached_urls = set()  # Set to store cached URLs

    def store_embeddings(self, embeddings, metadata):
        self.embeddings.append(embeddings)
        self.metadata.append(metadata)
        self.vector_store.add(embeddings, metadata)

    def retrieve_similar(self, query_embedding, top_k=5):
        similarities = np.dot(self.embeddings, query_embedding.T)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.metadata[i] for i in top_indices]

    def is_url_cached(self, url):
        return url in self.cached_urls

    def cache_url(self, url):
        self.cached_urls.add(url)