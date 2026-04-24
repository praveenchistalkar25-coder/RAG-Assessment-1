import os
import fitz  # PyMuPDF
import pandas as pd
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import os

def load_documents(folder_path):
    docs = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if fname.lower().endswith(".pdf"):
            docs.append({"type": "pdf", "path": fpath, "name": fname})
        elif fname.lower().endswith(".xlsx"):
            docs.append({"type": "excel", "path": fpath, "name": fname})
    return docs

def ocr_pdf(path):
    client = ComputerVisionClient(
        os.getenv("AZURE_OCR_ENDPOINT"),
        CognitiveServicesCredentials(os.getenv("AZURE_OCR_KEY"))
    )
    with open(path, "rb") as f:
        result = client.read_in_stream(f, raw=True)
    # Poll for results and extract text
    # (this SDK requires async polling)