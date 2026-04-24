# dataloading.py
import os
import fitz   # PyMuPDF for PDFs
import pandas as pd

def load_documents(folder_path: str) -> dict:
    """
    Scan a folder and load documents based on file type.
    - TXT: read directly
    - PDF: extract text with PyMuPDF
    - XLSX: flatten into text via pandas
    """
    docs = {}
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)

        if fname.lower().endswith(".txt"):
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                docs[fname] = f.read()

        elif fname.lower().endswith(".pdf"):
            text = ""
            with fitz.open(fpath) as pdf:
                for page in pdf:
                    text += page.get_text()
            docs[fname] = text

        elif fname.lower().endswith(".xlsx"):
            xls = pd.ExcelFile(fpath)
            text = ""
            for sheet in xls.sheet_names:
                df = pd.read_excel(fpath, sheet_name=sheet)
                text += df.to_string()
            docs[fname] = text

        else:
            print(f"⚠️ Skipping unsupported file type: {fname}")

    return docs
