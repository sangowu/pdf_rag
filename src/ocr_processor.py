import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import paddle
from paddleocr import PPStructureV3
import gc

from src.utils import load_config

config = load_config()
PDF_PATH = config.get("paths", {}).get("pdf_dir", "data/raw/OpenDataLab___OmniDocBench/pdfs")
OCR_OUTPUT_DIR = config.get("paths", {}).get("ocr_output_dir", "results/ocr_outputs")
OCR_STRUCTURED_DIR = config.get("paths", {}).get("ocr_structured_dir", "results/ocr_structured")
pdf_path = Path(PDF_PATH)
ocr_output_path = Path(OCR_OUTPUT_DIR)
markdown_path = ocr_output_path / "md"
json_path = ocr_output_path / "json"
ocr_structured_path = Path(OCR_STRUCTURED_DIR)

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._pipeline = None

    @property
    def pipeline(self) -> PPStructureV3:
        if self._pipeline is None:
            self._pipeline = PPStructureV3(lang="ch")
        return self._pipeline

    def _build_cache_path(self, pdf_path: str, cache_type: Literal["raw", "structured"]) -> Path:
        base = Path(pdf_path).stem
        if cache_type == "raw":
            return json_path / f"{base}_raw.json"
        if cache_type == "structured":
            return ocr_structured_path / f"{base}_structured.json"
        raise ValueError(f"Unsupported cache_type: {cache_type}")

    def _read_structured_content(self, pdf_path: str) -> Optional[List[dict]]:
        """Load structured cache if exists. Returns list of structured items or None."""
        cache_file = self._build_cache_path(pdf_path, "structured")
        if not cache_file.exists():
            return None
        logger.info("Using structured cache: %s", cache_file)
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    def _read_raw_content(self, pdf_path: str) -> Optional[List[dict]]:
        """Load raw OCR cache if exists. Returns list of raw page dicts or None."""
        cache_file = self._build_cache_path(pdf_path, "raw")
        if not cache_file.exists():
            return None
        logger.info("Using raw cache: %s", cache_file)
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [data]
        return data

    def _save_raw_cache(self, pdf_path: str, raw_pages: List[dict]) -> None:
        path = self._build_cache_path(pdf_path, "raw")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw_pages, f, ensure_ascii=False, indent=2)
        logger.info("Saved raw cache: %s", path)

    def _save_structured_cache(self, pdf_path: str, res_list: List[dict]) -> None:
        path = self._build_cache_path(pdf_path, "structured")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res_list, f, ensure_ascii=False, indent=2)
        logger.info("Saved structured cache: %s", path)

    def raw_to_structured_list(self, pdf_path: str, raw_pages: List[dict]) -> List[dict]:
        """
        Schema: convert raw OCR page list to structured list (one item per page).
        Does not read/write cache.
        """
        base_page_idx = self._extract_page_from_filename(pdf_path)
        return [
            self._build_structured_item(pdf_path, base_page_idx + i, page_data)
            for i, page_data in enumerate(raw_pages)
        ]

    def _extract_page_from_filename(self, pdf_path: str) -> int:
        filename = Path(pdf_path).name
        base = filename[:-4] if filename.lower().endswith('.pdf') else filename
        numbers = re.findall(r'\d+', base)
        return int(numbers[-1]) if numbers else 0

    def _build_structured_item(self, pdf_path: str, page_index: int, data: dict[str, Any]) -> dict[str, Any]:
        base = Path(pdf_path).stem.replace(".pdf", "")
        md_text = ""
        if markdown_path.exists():
            candidates = [p for p in markdown_path.glob("*.md") if base in p.stem]
            if candidates:
                parts = []
                for p in sorted(candidates):
                    with open(p, "r", encoding="utf-8") as f:
                        parts.append(f.read())
                md_text = "\n\n---\n\n".join(parts)
        else:
            markdown_path.mkdir(parents=True, exist_ok=True)
        if "res" in data and isinstance(data["res"], dict):
            data = data["res"]
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

    def process_pdf(self, pdf_path: str) -> List[dict]:
        """
        Pipeline: 1) structured cache -> return; 2) raw cache -> schema -> save structured -> return;
        3) OCR -> save raw -> schema -> save structured -> return.
        """
        ocr_output_path.mkdir(parents=True, exist_ok=True)
        ocr_structured_path.mkdir(parents=True, exist_ok=True)

        structured = self._read_structured_content(pdf_path)
        if structured is not None:
            return structured

        raw_pages = self._read_raw_content(pdf_path)
        if raw_pages is not None:
            res_list = self.raw_to_structured_list(pdf_path, raw_pages)
            self._save_structured_cache(pdf_path, res_list)
            return res_list

        output = self.pipeline.predict(input=pdf_path)

        raw_pages = []
        for res in output:
            data = res.json if isinstance(res.json, dict) else {}
            markdown = res.save_to_markdown(save_path=ocr_output_path / "md")
            raw_pages.append(data)
        self._save_raw_cache(pdf_path, raw_pages)

        res_list = self.raw_to_structured_list(pdf_path, raw_pages)
        self._save_structured_cache(pdf_path, res_list)

        gc.collect()
        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
        return res_list

    def list_pdf_files(
        self,
        directory: Optional[str] = None,
        pattern: str = "*.pdf",
    ) -> List[str]:
        """
        List PDF file paths under directory. If directory is None, use config pdf_dir.
        Returns list of absolute path strings.
        """
        dir_path = Path(directory) if directory else pdf_path
        if not dir_path.is_dir():
            logger.warning("Directory does not exist: %s", dir_path)
            return []
        out = [str(p.absolute()) for p in dir_path.glob(pattern) if p.is_file()]
        return sorted(out)

    def process_batch(
        self,
        pdf_paths: Optional[List[str]] = None,
        directory: Optional[str] = None,
    ) -> Dict[str, List[dict]]:
        """
        Process multiple PDFs. If pdf_paths is None, list from directory (or config pdf_dir).
        Returns mapping: pdf_path -> list of structured items per page.
        """
        if pdf_paths is None:
            pdf_paths = self.list_pdf_files(directory=directory)
        results = {}
        for path in pdf_paths:
            try:
                results[path] = self.process_pdf(path)
            except Exception as e:
                logger.exception("Failed to process %s: %s", path, e)
                raise
        return results

    def extract_content(self, result_obj: dict) -> str:
        logger.debug("Extracting content from result object with keys: %s", list(result_obj.keys()))
        parsing_list = result_obj.get("core_content", {}).get("parsing_res_list", result_obj.get("parsing_res_list", []))
        page_text = ""
        for block in parsing_list:
            block_dict = block.todict() if hasattr(block, "todict") else (block if isinstance(block, dict) else {})
            if not isinstance(block_dict, dict):
                continue
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    ocr_processor = OCRProcessor(config)
    pdf_list = ocr_processor.list_pdf_files()
    if not pdf_list:
        logger.warning("No PDF files found in %s", pdf_path)
    else:
        first_file = pdf_list[0]
        logger.info("Using first file: %s", first_file)
        image_pdf = ocr_processor.process_pdf(first_file)
        all_keys = set()
        for item in image_pdf:
            all_keys.update(item.keys())
        logger.info("Result keys sample: %s", all_keys if all_keys else "No keys found")

