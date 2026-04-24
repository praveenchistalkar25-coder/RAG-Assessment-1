"""
embedding.py — Embedding generation module.

Enhancements:
- Clear docstrings and reasoning comments.
- Robust error handling with explicit logging.
- Normalization ensures cosine similarity produces meaningful scores.
"""

import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------------------------------------------
# Environment setup
# -------------------------------------------------------------------
# Load environment variables from .env file (keeps secrets out of codebase).
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Check your .env file.")

# Initialize OpenAI client once per module
client = OpenAI(api_key=api_key)

# -------------------------------------------------------------------
# Embedding function
# -------------------------------------------------------------------

def embed_text(text: str) -> np.ndarray:
    """
    Generate normalized embedding for given text with logging and error handling.

    Reasoning:
    - Normalization: cosine similarity requires unit-length vectors to produce
      consistent relevance scores. Without normalization, magnitude differences
      distort similarity.
    - Error handling: explicit logging ensures ingestion failures are visible
      instead of silent.
    - dtype=np.float32: reduces memory footprint and ensures compatibility with
      vector databases like PGVector or Pinecone.
    """
    try:
        # Call OpenAI embedding API
        resp = client.embeddings.create(
            model="text-embedding-3-small",  # chosen for cost-efficiency and sufficient fidelity
            input=text
        )

        # Convert to numpy array
        emb = np.array(resp.data[0].embedding, dtype=np.float32)

        # Normalize for cosine similarity
        norm = np.linalg.norm(emb)
        if norm == 0:
            raise ValueError("Generated embedding has zero norm (invalid vector).")
        emb = emb / norm

        # Debug logging (optional: can be toggled via env var)
        print(f"[DEBUG] Embedding generated for text: '{text[:50]}...' (dim={len(emb)})")

        return emb

    except Exception as e:
        # Explicit error logging
        print(f"[ERROR] Failed to generate embedding: {e}")
        raise



