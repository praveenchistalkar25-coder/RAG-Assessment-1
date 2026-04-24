"""
App.py — Retrieval-Augmented Generation (RAG) pipeline entry point.

Enhancements:
- Clear structure: ingestion → retrieval → reranking → synthesis.
- Reasoning comments: explain *why* each design choice was made.
- Readability: consistent naming, concise logging, loop mode for multiple queries.
"""

import os
from dataloading import load_documents
from chunking import chunk_document
from embedding import embed_text
from retrival import LocalVectorStore, relevance_bar
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

# -------------------------------------------------------------------
# Environment setup
# -------------------------------------------------------------------
# Load environment variables from .env file (keeps secrets out of codebase).
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cross-encoder reranker:
# Reasoning: cosine similarity is fast but coarse. Cross-encoder evaluates
# query-chunk pairs directly, producing finer-grained relevance scores.
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Configurable constants
FOLDER = r"C:\Users\Swamini\Downloads\Assessment Data"
TOP_K = 10  # retrieve more chunks initially, then rerank down to top 5

# -------------------------------------------------------------------
# Answer synthesis
# -------------------------------------------------------------------
def answer_query(query, results):
    """
    Use LLM to synthesize a final answer grounded in retrieved chunks.

    Reasoning:
    - Include metadata (source, section) so the LLM can cite explicitly.
    - Temperature kept low (0.2) to prioritize factual synthesis over creativity.
    """
    context = "\n\n".join([
        f"Source: {item['source']} | Section: {item['section_title']}\nChunk: {item['chunk']}"
        for _, item in results
    ])

    prompt = f"""
You are an AI assistant. Use the following context to answer the query.
Query: {query}

Context:
{context}

Answer (cite sources explicitly):
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",   # or gpt-4o / gpt-4-turbo depending on your access
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

# -------------------------------------------------------------------
# Reranking
# -------------------------------------------------------------------
def rerank_results(query, results):
    """
    Rerank retrieved chunks using cross-encoder scores.

    Reasoning:
    - Cosine similarity gives a first-pass ranking.
    - Cross-encoder refines ranking by evaluating query-chunk pairs directly.
    - Scores can be negative, so we normalize to 0–100 for user-friendly display.
    """
    pairs = [(query, item['chunk']) for _, item in results]
    scores = reranker.predict(pairs)

    min_score, max_score = min(scores), max(scores)
    normalized = [
        (round(((score - min_score) / (max_score - min_score)) * 100, 2), item)
        for score, (_, item) in zip(scores, results)
    ]

    reranked = sorted(normalized, key=lambda x: x[0], reverse=True)
    return reranked

# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
def main():
    # Ingest documents once per run
    docs = load_documents(FOLDER)
    store = LocalVectorStore()

    # Chunk + embed each document
    # Reasoning: chunking preserves context granularity, embeddings enable semantic search.
    for doc in docs:
        chunks = chunk_document(doc)
        print(f"Processing {doc['name']} → {len(chunks)} chunks extracted")
        for c in chunks:
            emb = embed_text(c["chunk"])
            store.add(
                c["chunk"], emb, doc["name"],
                c["doc_type"], c["section_title"], c["chunk_id"]
            )

    # Loop mode: allows multiple queries without restarting
    while True:
        query = input("\nQuery (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting...")
            break

        query_emb = embed_text(query)
        results = store.search(query_emb, top_k=TOP_K, min_relevance=0)

        if not results:
            print("No relevant chunks found.\n")
            continue

        # Rerank results for finer relevance
        results = rerank_results(query, results)

        print("\n=== Top Relevant Chunks (Cross-Encoder Reranked) ===\n")
        for relevance, item in results[:5]:  # show top 5 after reranking
            print(f"Source: {item['source']} | Section: {item['section_title']} | ID: {item['chunk_id']}")
            print(relevance_bar(relevance))
            print(f"Chunk: {item['chunk'][:500]}...\n")

        # Final grounded answer
        final_answer = answer_query(query, results[:5])
        print("\n=== Final Answer ===\n")
        print(final_answer)

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
