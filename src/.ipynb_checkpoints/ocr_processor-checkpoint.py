import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
import json, re
from typing import Any, Dict, Literal
from PIL import Image
from pathlib import Path
from paddleocr import PPStructureV3
from transformers import data

from src.utils import load_config

config = load_config()
PDF_PATH = config.get("paths", {}).get("pdf_dir", "data/raw/OpenDataLab___OmniDocBench/pdfs")
OCR_OUTPUT_DIR = config.get("paths", {}).get("ocr_output_dir", "results/ocr_outputs")
OCR_CACHE_DIR = config.get("paths", {}).get("ocr_cache_dir", "results/ocr_cache")
pdf_path = Path(PDF_PATH)
ocr_output_path = Path(OCR_OUTPUT_DIR)
ocr_cache_path = Path(OCR_CACHE_DIR)
FIRST_FILE = next((f for f in pdf_path.glob("*.pdf") if f.is_file()), None)
if FIRST_FILE:
    FIRST_FILE = str(FIRST_FILE.absolute())   # 转成绝对路径
    print("Using absolute path:", FIRST_FILE)

class OCRProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config 

    def _build_cache_path(self, pdf_path: str, cache_type: Literal["raw", "structured"]) -> Path:
        base = Path(pdf_path).stem  # 保留原文件名基底
        if cache_type == "raw":
            return ocr_output_path / f"{base}_raw.json"
        if cache_type == "structured":
            return ocr_cache_path / f"{base}_structured.json"
        raise ValueError(f"Unsupported cache_type: {cache_type}")

    def _load_cache(self, pdf_path: str, cache_type: Literal["raw", "structured"]) -> str | None:
        cache_file = self._build_cache_path(pdf_path, cache_type)
        if cache_file.exists():
            print(f"Using {cache_type} cache: {cache_file}")
            return str(cache_file)
        return None
    
    def _extract_page_from_filename(self, pdf_path) -> int:
        filename = Path(pdf_path).name
        base = filename[:-4] if filename.lower().endswith('.pdf') else filename
        numbers = re.findall(r'\d+', base)
        return int(numbers[-1]) if numbers else 0

    def _build_structured_item(self, pdf_path: str, page_index: int, data: dict[str, Any]) -> dict[str, Any]:
        md_text = ""
        md_val = data.get("markdown", "")
        if isinstance(md_val, dict):
            md_text = md_val.get("markdown_texts", "") or ""
        elif isinstance(md_val, str):
            md_text = md_val

        return {
            "filename": os.path.basename(pdf_path),
            "page_index": page_index,
            "core_content": {
                "parsing_res_list": data.get("parsing_res_list", []),
                "markdown": md_text,
                "tables": data.get("table_res_list", []),
                "formulas": data.get("formula_res_list", []),
                "seals": data.get("seal_res_list", []),
            },
            "metadata": {
                "width": data.get("width", 0),
                "height": data.get("height", 0),
                "doc_preprocessor_res": data.get("doc_preprocessor_res", {}),
                "layout_det_res": data.get("layout_det_res", {}),
                "region_det_res": data.get("region_det_res", {}),
                "overall_ocr_res": data.get("overall_ocr_res", {}),
                "table_res_list": data.get("table_res_list", []),
                "seal_res_list": data.get("seal_res_list", []),
                "chart_res_list": data.get("chart_res_list", []),
                "formula_res_list": data.get("formula_res_list", []),
                "imgs_in_doc": data.get("imgs_in_doc", []),
                "model_settings": data.get("model_settings", {}),
            },
        }

    def process_pdf(self, pdf_path: str) -> list[dict]:
        ocr_output_path.mkdir(parents=True, exist_ok=True)
        ocr_cache_path.mkdir(parents=True, exist_ok=True)

        structured_cache = self._load_cache(pdf_path, cache_type="structured")
        if structured_cache:
            with open(structured_cache, "r", encoding="utf-8") as f:
                cached_structured = json.load(f)
            return cached_structured if isinstance(cached_structured, list) else [cached_structured]

        # 2) 再读 raw 缓存
        raw_cache = self._load_cache(pdf_path, cache_type="raw")
        if raw_cache:
            with open(raw_cache, "r", encoding="utf-8") as f:
                raw_pages = json.load(f)

            if isinstance(raw_pages, dict):
                raw_pages = [raw_pages]

            res_list: list[dict[str, Any]] = []
            base_page_idx = self._extract_page_from_filename(pdf_path)
            for i, data in enumerate(raw_pages):
                res_list.append(self._build_structured_item(pdf_path, base_page_idx + i, data))

            # 回写 structured 缓存
            structured_path = self._build_cache_path(pdf_path, "structured")
            with open(structured_path, "w", encoding="utf-8") as f:
                json.dump(res_list, f, ensure_ascii=False, indent=2)

            return res_list

        pipeline = PPStructureV3(lang="ch")
        output = pipeline.predict(input=pdf_path)

        raw_pages: list[dict[str, Any]] = []
        res_list: list[dict[str, Any]] = []
        base_page_idx = self._extract_page_from_filename(pdf_path)

        for i, res in enumerate(output):
            data = res.json if isinstance(res.json, dict) else {}
            raw_pages.append(data)
            res_list.append(self._build_structured_item(pdf_path, base_page_idx + i, data))

        # 保存 raw 缓存
        raw_path = self._build_cache_path(pdf_path, "raw")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_pages, f, ensure_ascii=False, indent=2)

        # 保存 structured 缓存
        structured_path = self._build_cache_path(pdf_path, "structured")
        with open(structured_path, "w", encoding="utf-8") as f:
            json.dump(res_list, f, ensure_ascii=False, indent=2)

        return res_list

    def extract_content(self, result_obj: dict) -> str:
        print(f"Extracting content from result object with keys: {result_obj.keys()}")
        parsing_list = result_obj.get('parsing_res_list', [])
        page_text = ""
        for block in parsing_list:
            print(f"Processing block: {dir(block)}")
            block_dict = block.todict()  
            print(f"Block's keys: {block_dict.keys()}")
            label = block_dict.get('block_label', 'unknown')
            content = block_dict.get('block_content', '')
            if label == 'paragraph_title':
                page_text += f"\n## {content}\n"
            elif label == 'table':
                page_text += f"[TABLE_START]\n{content}\n[TABLE_END]\n"
            else:
                page_text += f"{content}\n"
        return page_text

if __name__ == "__main__":
    ocr_processor = OCRProcessor(config)
    image_pdf = ocr_processor.process_pdf(FIRST_FILE)
    # print(f"Processing file: {type(image_pdf)}")
    print('-'*100)
    all_keys = set()
    for item in image_pdf:
        all_keys.update(item.keys())
    print(f"结果列表的内容示例: {all_keys if all_keys else 'No keys found'}")
    print('-'*100)
    # parsing_res_list = image_pdf.get('parsing_res_list')
    # if parsing_res_list and len(parsing_res_list) > 0:
    #     results = []
    #     for block in parsing_res_list:
    #         results.append({
    #             "block_label": block.get('block_label', 'unknown'),
    #             "block_content": block.get('block_content', '')
    #         })

    #     print(f"块的内容示例: {results if results else 'No blocks found'}")
    # print(f"Processing complete for file: {image_pdf[0]}")
    print("-" * 10)
    print("-" * 10)

