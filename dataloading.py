"""
dataloading.py — Document loader for PDFs and Excel files.

Enhancements:
- Clear docstrings and reasoning comments.
- Modular design: separates file detection from OCR logic.
- Consistent metadata structure for downstream chunking.
"""

import os
import fitz  # PyMuPDF for PDF parsing
import pandas as pd
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

# -------------------------------------------------------------------
# Document loader
# -------------------------------------------------------------------

def load_documents(folder_path):
    """
    Scan a folder and return a list of documents with metadata.

    Reasoning:
    - Keeps ingestion modular: only identifies file type and path.
    - Actual parsing/chunking is delegated to chunking.py.
    - Metadata includes type, path, and name for traceability.
    """
    docs = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)

        # Detect supported file types
        if fname.lower().endswith(".pdf"):
            docs.append({"type": "pdf", "path": fpath, "name": fname})
        elif fname.lower().endswith(".xlsx"):
            docs.append({"type": "excel", "path": fpath, "name": fname})

        # Future extension: add DOCX, TXT, HTML handlers here

    return docs


# -------------------------------------------------------------------
# OCR fallback (legacy ComputerVision SDK)
# -------------------------------------------------------------------

def ocr_pdf(path):
    """
    OCR fallback for image-only PDFs using Azure Computer Vision (legacy SDK).

    Reasoning:
    - Some PDFs contain only scanned images (no extractable text).
    - This function uses the older ComputerVisionClient for OCR.
    - Note: This SDK requires async polling to retrieve results.
    - Consider migrating to azure.ai.vision.imageanalysis for consistency.
    """
    client = ComputerVisionClient(
        os.getenv("AZURE_OCR_ENDPOINT"),
        CognitiveServicesCredentials(os.getenv("AZURE_OCR_KEY"))
    )

    with open(path, "rb") as f:
        # Submit OCR request
        result = client.read_in_stream(f, raw=True)

    # TODO: Implement polling loop to wait for OCR results
    # Example:
    # while True:
    #     status = client.get_read_result(result.headers["Operation-Location"])
    #     if status.status not in ["running", "notStarted"]:
    #         break
    # return " ".join([line.text for page in status.analyze_result.read_results
    #                  for line in page.lines])

    return ""  # placeholder until polling implemented
