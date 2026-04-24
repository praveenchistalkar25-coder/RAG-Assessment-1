import os
from dataloading import load_documents
from chunking import chunk_document
from embedding import embed_text
from retrival import LocalVectorStore

FOLDER = r"C:\Users\Swamini\Downloads\Assessment Data"
QUERY = "AI solutions delivered for healthcare clients"

def main():
    docs = load_documents(FOLDER)
    store = LocalVectorStore()

    # Ingest and embed with metadata
    for doc in docs:
        chunks = chunk_document(doc)
        for c in chunks:
            emb = embed_text(c["chunk"])
            store.add(
                c["chunk"], emb, doc["name"],
                c["doc_type"], c["section_title"], c["chunk_id"]
            )

    # Query
    query_emb = embed_text(QUERY)
    results = store.search(query_emb, top_k=3)

    # Print results with metadata
    for score, item in results:
        print(f"Source: {item['source']} | Type: {item['doc_type']} | Section: {item['section_title']} | ID: {item['chunk_id']}")
        print(f"Chunk: {item['chunk'][:300]}...\n")
        print("# Retrieval rationale: metadata ensures auditability, deduplication, and context preservation.\n")

if __name__ == "__main__":
    main()
