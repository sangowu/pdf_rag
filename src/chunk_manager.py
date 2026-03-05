from pathlib import Path
import json
from src.utils import load_config
import logging
config = load_config()
OCR_STRUCTURED_DIR = config.get("paths", {}).get("ocr_structured_dir", "results/ocr_structured")
ocr_structured_path = Path(OCR_STRUCTURED_DIR)
CHUNK_SCHEMA_DIR = config.get("paths", {}).get("chunk_results_dir", "results/chunk_results")
chunk_schema_path = Path(CHUNK_SCHEMA_DIR)
logger = logging.getLogger(__name__)

class ChunkManager:
    def __init__(self):
        chunk_cfg = config.get("chunking", {})
        self.chunk_size = chunk_cfg.get("chunk_size", 512)
        self.chunk_overlap = chunk_cfg.get("chunk_overlap", 128)
        self.chunk_type = chunk_cfg.get("type", "fixed")
        self.separators = chunk_cfg.get("separators", ["\n\n", "\n", "。", "，", "、", ",", ".", ""])
        self.keep_separator = chunk_cfg.get("keep_separator", False)

    def list_full_paths(self,directory: str, pattern: str = "*", limit: int | None = None):
        path = Path(directory)
        if not path.is_dir():
            return []
        files = sorted(path.glob(pattern), key=lambda p: p.name)
        files = [p for p in files if p.is_file()]
        if limit is not None:
            files = files[:limit]
        return [str(p.absolute()) for p in files]

    def _read_ocr_json(self, ocr_json_path):
        with open(ocr_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _write_chunk_schema(self, chunk_schema, chunk_schema_path):
        with open(chunk_schema_path, "w", encoding="utf-8") as f:
            json.dump(chunk_schema, f, ensure_ascii=False, indent=4)

    def _write_all_chunk(self, chunk_schema_list, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunk_schema_list, f, ensure_ascii=False, indent=4)

    def _split_text_by_chunk_size(self, text):
        chunks = []
        chunk_index = 0
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append({
                "chunk_index": chunk_index,
                "chunks_content": text[start:end],
                "char_count": len(text[start:end]),
            })
            if self.chunk_overlap < self.chunk_size:
                start += self.chunk_size - self.chunk_overlap
                chunk_index += 1
            else:
                start += self.chunk_size
                chunk_index += 1
        return chunks

    def _split_text_recursive(self, text):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator=self.keep_separator,
        )
        raw = splitter.split_text(text)
        return [
            {"chunk_index": i, "chunks_content": s, "char_count": len(s)}
            for i, s in enumerate(raw)
        ]
    
    def _generate_chunk_id(self, file_name, page_index, chunk_index):
        return f"{file_name}_p{page_index}_c{chunk_index}"

    def generate_chunks(self, ocr_json_path):
        data = self._read_ocr_json(ocr_json_path)
        chunk_schema_path.mkdir(parents=True, exist_ok=True)
        chunk_schema_list = []
        for page_item in data:
            filename = page_item["filename"]
            stem = Path(filename).stem
            core_content = page_item["core_content"]
            page_index = page_item["page_index"]

            parsing_res_list = core_content["parsing_res_list"]
            parsing_res_list_sorted = sorted(parsing_res_list, key=lambda x: x.get("block_id", 0))
            block_content_all = ""

            for parsing_res in parsing_res_list_sorted:
                if parsing_res.get("block_label") != "text":
                    block_content_all += f"\n##{parsing_res.get('block_label')}\n"
                else:
                    block_content_all += "\n"
                block_content_all += parsing_res.get("block_content", "")
                
            if self.chunk_type == "fixed":
                chunks = self._split_text_by_chunk_size(block_content_all)
            elif self.chunk_type == "recursive":
                chunks = self._split_text_recursive(block_content_all)
            else:
                raise ValueError(f"Invalid chunk type: {self.chunk_type}")
            for chunk in chunks:
                chunk_id = self._generate_chunk_id(stem, page_index, chunk["chunk_index"])
                chunk_schema={
                    "file_name": stem,
                    "page_index": page_index,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["chunks_content"],
                    "char_count": chunk["char_count"],
                }
                chunk_schema_list.append(chunk_schema)
        for chunk_s in chunk_schema_list:
            out_path = chunk_schema_path / f"{chunk_s['file_name']}_{chunk_s['page_index']}_{chunk_s['chunk_index']}.json"
            self._write_chunk_schema(chunk_s, out_path)
            logger.info("Chunk schema generated and saved to %s", out_path)

        return chunk_schema_list

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )
    cm = ChunkManager()
    file_list = cm.list_full_paths(ocr_structured_path, "*.json", limit=50)
    all_chunk = []
    for file_path in file_list:
        all_chunk.extend(cm.generate_chunks(file_path))
    cm._write_all_chunk(all_chunk, chunk_schema_path / "all_chunks.json")