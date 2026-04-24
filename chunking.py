import os
import fitz
import pandas as pd
import hashlib
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential

def generate_chunk_id(text, source):
    """Generate a stable hash ID for deduplication and tracking."""
    return hashlib.sha256((source + text).encode("utf-8")).hexdigest()[:12]

def split_text(text, max_len=500):
    """Split long text into smaller segments for better retrieval granularity."""
    words = text.split()
    for i in range(0, len(words), max_len):
        yield " ".join(words[i:i+max_len])

def ocr_pdf(path):
    """OCR fallback for image-only PDFs using Azure Vision."""
    client = ImageAnalysisClient(
        endpoint=os.getenv("AZURE_OCR_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_OCR_KEY"))
    )
    with open(path, "rb") as f:
        result = client.analyze(image_data=f, visual_features=["Read"])
    return " ".join([line.text for block in result.read.blocks for line in block.lines])

def chunk_pdf(path, source):
    """Extract chunks from PDF with block-level text and OCR fallback."""
    doc = fitz.open(path)
    chunks = []
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        if not blocks:
            text = ocr_pdf(path)
        else:
            text = " ".join([block[4] for block in blocks if block[4].strip()])
        # Split into smaller segments
        for seg in split_text(text, max_len=500):
            chunk_id = generate_chunk_id(seg, source)
            chunks.append({
                "chunk": seg,
                "doc_type": "pdf",
                "section_title": f"Page {i+1}",
                "chunk_id": chunk_id
            })
    return chunks

def chunk_excel(path, source):
    """Extract chunks from all sheets in Excel file."""
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
                    "chunk_id": chunk_id
                })
    return chunks

def chunk_document(doc):
    """Dispatch chunking based on document type."""
    if doc["type"] == "pdf":
        return chunk_pdf(doc["path"], doc["name"])
    elif doc["type"] == "excel":
        return chunk_excel(doc["path"], doc["name"])
    else:
        return []

