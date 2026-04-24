# RAG-ASSESSMENT

Retrieval-Augmented Generation (RAG) pipeline for document ingestion, chunking, embedding, retrieval, reranking, and answer synthesis.

---

## 📂 Project Structure

RAG-ASSESSMENT/
│── app.py              # Entry point: orchestrates ingestion → retrieval → reranking → synthesis
│── chunking.py         # Splits documents into smaller chunks, with OCR fallback for PDFs
│── dataloading.py      # Loads documents from folder, detects file type (PDF/Excel)
│── embedding.py        # Generates normalized embeddings using OpenAI API
│── retrival.py         # Local in-memory vector store with cosine similarity search
│── .env                # Environment variables (API keys, endpoints) — ignored in Git
│── .gitignore          # Ensures secrets and cache files are not committed
│── pycache/        # Python cache (ignored)


#Code : 
---

## ⚙️ Pipeline Overview

1. **Document Loading (`dataloading.py`)**
   - Scans a folder for supported file types (`.pdf`, `.xlsx`).
   - Returns metadata (`type`, `path`, `name`) for downstream processing.
   - OCR fallback available for image-only PDFs (Azure Computer Vision).

2. **Chunking (`chunking.py`)**
   - **PDFs**: Extract text blocks via PyMuPDF; fallback to Azure OCR if no text.
   - **Excel**: Flatten rows into text strings, split into segments.
   - Adds metadata: `doc_type`, `section_title`, `chunk_id`, `parser`.

3. **Embedding (`embedding.py`)**
   - Uses OpenAI `text-embedding-3-small` model.
   - Normalizes vectors for cosine similarity.
   - Error handling + debug logging for transparency.

4. **Retrieval (`retrival.py`)**
   - Local in-memory vector store.
   - Cosine similarity search returns top‑K chunks with relevance scores.
   - Visual relevance bar (`█░`) for interpretability.

5. **Reranking (`app.py`)**
   - Cross‑encoder (`ms-marco-MiniLM-L-6-v2`) refines ranking.
   - Normalizes scores to 0–100% for user-friendly display.
   - Ensures query-dependent relevance beyond cosine similarity.

6. **Answer Synthesis (`app.py`)**
   - LLM (`gpt-4o-mini`) generates final answer grounded in retrieved chunks.
   - Includes metadata (source, section) for explicit citations.
   - Low temperature (0.2) for factual consistency.

---

## 🚀 How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt

Suggested packages:

openai

numpy

pandas

pymupdf

python-dotenv

sentence-transformers

azure-ai-vision (for OCR)


**Set environment variables
Create a .env file:
Code: 
OPENAI_API_KEY=your_openai_key
AZURE_OCR_ENDPOINT=https://your-ocr-endpoint.cognitiveservices.azure.com/
AZURE_OCR_KEY=your_azure_key

**Run the app
python app.py


**Query interactively
Query (or type 'exit' to quit): AI solutions we have delivered for healthcare clients
