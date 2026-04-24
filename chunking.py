"""
chunking.py — Document chunking logic for PDFs and Excel files.

Enhancements:
- Clear function docstrings with reasoning.
- Consistent naming and structure.
- Comments explain *why* each design choice was made (e.g., OCR fallback, hash IDs).
"""

import os
import fitz  # PyMuPDF for PDF parsing
import pandas as pd
import hashlib
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def generate_chunk_id(text, source):
    """
    Generate a stable hash ID for deduplication and tracking.

    Reasoning:
    - Hashing ensures identical chunks across runs/documents get the same ID.
    - Using both source + text prevents collisions between different files.
    - Truncate to 12 chars for readability while remaining unique enough.
    """
    return hashlib.sha256((source + text).encode("utf-8")).hexdigest()[:12]


def split_text(text, max_len=500):
    """
    Split long text into smaller segments for better retrieval granularity.

    Reasoning:
    - Embedding models perform better on shorter segments.
    - 500 words chosen as a balance: enough context, but not too large for embedding.
    - Yields segments so caller can iterate and build chunks.
    """
    words = text.split()
    for i in range(0, len(words), max_len):
        yield " ".join(words[i:i+max_len])


# -------------------------------------------------------------------
# OCR fallback
# -------------------------------------------------------------------

def ocr_pdf(path):
    """
    OCR fallback for image-only PDFs using Azure Vision.

    Reasoning:
    - Some PDFs contain only scanned images (no extractable text).
    - Azure OCR ensures we still capture text content for retrieval.
    - Returns concatenated text from all detected lines.
    """
    client = ImageAnalysisClient(
        endpoint=os.getenv("AZURE_OCR_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_OCR_KEY"))
    )
    with open(path, "rb") as f:
        result = client.analyze(image_data=f, visual_features=["Read"])
    return " ".join([line.text for block in result.read.blocks for line in block.lines])


# -------------------------------------------------------------------
# Chunking logic per file type
# -------------------------------------------------------------------

def chunk_pdf(path, source):
    """
    Extract chunks from PDF with block-level text and OCR fallback.

    Reasoning:
    - Use PyMuPDF to extract text blocks (preserves layout better than raw text).
    - If no text blocks found, fallback to OCR for image-only PDFs.
    - Each page is split into smaller segments (500 words) for embedding.
    - Metadata includes doc_type, section_title, and chunk_id for traceability.
    """
    doc = fitz.open(path)
    chunks = []
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        if not blocks:
            text = ocr_pdf(path)
            parser_used = "AzureOCR"
        else:
            text = " ".join([block[4] for block in blocks if block[4].strip()])
            parser_used = "PyMuPDF"

        # Split into smaller segments
        for seg in split_text(text, max_len=500):
            chunk_id = generate_chunk_id(seg, source)
            chunks.append({
                "chunk": seg,
                "doc_type": "pdf",
                "section_title": f"Page {i+1}",
                "chunk_id": chunk_id,
                "parser": parser_used  # track which parser was used
            })
    return chunks


def chunk_excel(path, source):
    """
    Extract chunks from all sheets in Excel file.

    Reasoning:
    - Each row is treated as a logical unit of information.
    - Flatten row values into a single string, skipping NaN values.
    - Split long rows into smaller segments for embedding.
    - Metadata includes sheet name and row number for traceability.
    """
    chunks = []
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        for i, row in df.iterrows():
            text = " ".join([str(x) for x in row.values if pd.notna(x)])
            for seg in split_text(text, max_len=500):
                chunk_id = generate_chunk_id(seg, source)
                chunks.append({
                    "chunk": seg,
                    "doc_type": "excel",
                    "section_title": f"{sheet} Row {i+1}",
                    "chunk_id": chunk_id,
                    "parser": "PandasExcel"  # track parser used
                })
    return chunks


# -------------------------------------------------------------------
# Dispatcher
# -------------------------------------------------------------------

def chunk_document(doc):
    """
    Dispatch chunking based on document type.

    Reasoning:
    - Keeps logic modular: each file type has its own chunking strategy.
    - Easy to extend: add new handlers for DOCX, TXT, HTML, etc.
    """
    if doc["type"] == "pdf":
        return chunk_pdf(doc["path"], doc["name"])
    elif doc["type"] == "excel":
        return chunk_excel(doc["path"], doc["name"])
    else:
        return []

