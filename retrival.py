"""
retrival.py — Local in-memory vector store for chunk retrieval.

Enhancements:
- Clear docstrings and reasoning comments.
- Explicit explanation of cosine similarity and normalization.
- Visual relevance bar for interpretability.
"""

import numpy as np

# -------------------------------------------------------------------
# Local Vector Store
# -------------------------------------------------------------------

class LocalVectorStore:
    """
    A simple in-memory vector store for document chunks.

    Reasoning:
    - Provides a lightweight alternative to external vector DBs (e.g., PGVector, Pinecone).
    - Useful for prototyping and validation before scaling to enterprise-grade storage.
    - Stores both embeddings and metadata for traceability.
    """

    def __init__(self):
        # Internal list of chunks with metadata and embeddings
        self.store = []

    def add(self, chunk, embedding, source, doc_type=None, section_title=None, chunk_id=None):
        """
        Add a chunk with metadata into the store.

        Reasoning:
        - Each chunk is stored with its embedding and metadata (source, type, section, ID).
        - Metadata ensures auditability: we can trace back answers to original documents.
        """
        self.store.append({
            "chunk": chunk,
            "embedding": embedding,
            "source": source,
            "doc_type": doc_type,
            "section_title": section_title,
            "chunk_id": chunk_id
        })

    def search(self, query_embedding, top_k=3, min_relevance=0):
        """
        Search for top_k most relevant chunks using cosine similarity.

        Reasoning:
        - Cosine similarity measures angle between vectors, not magnitude.
        - Normalization ensures embeddings are unit-length, so dot product ≈ cosine similarity.
        - Convert similarity to percentage (0–100) for interpretability.
        - min_relevance filter hides weak matches, improving answer quality.
        """
        scored = []
        for item in self.store:
            # Cosine similarity = dot(u, v) / (||u|| * ||v||)
            score = np.dot(item["embedding"], query_embedding) / (
                np.linalg.norm(item["embedding"]) * np.linalg.norm(query_embedding)
            )
            relevance = round(score * 100, 2)  # convert to percentage

            if relevance >= min_relevance:
                scored.append((relevance, item))

        # Sort by relevance descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top_k results
        return scored[:top_k]


# -------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------

def relevance_bar(relevance, length=20):
    """
    Return a bar visualization for relevance percentage.

    Reasoning:
    - Humans interpret visual bars faster than raw numbers.
    - Filled blocks (█) represent strength of match.
    - Empty blocks (░) represent gap to 100%.
    - Length=20 chosen for readability in console output.
    """
    filled = int((relevance / 100) * length)
    return "[" + "█" * filled + "░" * (length - filled) + f"] {relevance}%"




