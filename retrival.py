import numpy as np

class LocalVectorStore:
    def __init__(self):
        self.store = []

    def add(self, chunk, embedding, source, doc_type=None, section_title=None, chunk_id=None):
        """Add a chunk with metadata into the store."""
        self.store.append({
            "chunk": chunk,
            "embedding": embedding,
            "source": source,
            "doc_type": doc_type,
            "section_title": section_title,
            "chunk_id": chunk_id
        })

    def search(self, query_embedding, top_k=3):
        scored = []
        for item in self.store:
            score = np.dot(item["embedding"], query_embedding) / (
                np.linalg.norm(item["embedding"]) * np.linalg.norm(query_embedding)
            )
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


