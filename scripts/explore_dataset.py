import json
import pandas as pd
import os
import PyPDF2
from typing import Any, List, Dict
from src.utils import load_config

config = load_config()

paths = config.get("paths", {})
GT_PATH = paths.get("omnidoc_json", "data/raw/OpenDataLab___OmniDocBench/OmniDocBench.json")
PDF_PATH = paths.get("pdf_dir", "data/raw/OpenDataLab___OmniDocBench/pdfs")


# chunk_size = config.get("chunking", {}).get("chunk_size", 512)

# Use encoding="utf-8" to avoid UnicodeDecodeError on Windows
# with open(GT_PATH, "r", encoding="utf-8") as f:
#     df = pd.DataFrame(json.load(f))


def list_pdf_files(input_dir: str) -> list[str]:

    pdf_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def inspect_pdf(pdf_path: str) -> Dict[str, Any]:
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return {
                "num_pages": len(reader.pages),
                "file_size": os.path.getsize(pdf_path)
            }
    except Exception:
        return print(f"Error reading {pdf_path}")

def inspect_omnidoc_json(json_path) -> pd.DataFrame:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return pd.DataFrame(data)
    except Exception:
        return print(f"Error reading {json_path}")





if __name__ == "__main__":

    data = list_pdf_files(PDF_PATH)
    pdf = data[0]

    print(inspect_pdf(pdf))
    print(inspect_omnidoc_json(GT_PATH).head(2))