import os
from openai import OpenAI
import chromadb
from src.utils import load_config
import json
config = load_config()
import logging
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.all_chunk_path = config.get("paths", {}).get("all_chunk_path")
        self.batch_size = config.get("embedding", {}).get("batch_size", 32)
        self._chroma_client = None
        self._chroma_collection = None
        self.top_k = config.get("evaluation", {}).get("top_k", 5)

    def _init_chroma_client(self):
        if self._chroma_collection is not None:
            return self._chroma_collection
        chroma_cfg = config.get("chromadb", {})
        persist_dir = chroma_cfg.get("persist_directory", "vectors/chroma_db")
        collection_name = chroma_cfg.get("collection_name", "pdf_chunks")
        distance_fn = chroma_cfg.get("distance_fn", "cosine")
        self._chroma_client = chromadb.PersistentClient(path=persist_dir)
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine" if distance_fn == "cosine" else "l2"},
        )
        return self._chroma_collection

    def _chunk_row_to_chroma_metadata(self, row: dict) -> dict:
        return {
            "file_name": str(row.get("file_name", "")),
            "page_index": int(row.get("page_index", 0)),
            "chunk_index": int(row.get("chunk_index", 0)),
            "char_count": int(row.get("char_count", 0)),
        }

    def add_chunks_to_chroma(self, new_table: list[dict], batch_size: int = 100) -> None:
        if not new_table:
            return
        coll = self._init_chroma_client()
        for i in range(0, len(new_table), batch_size):
            batch = new_table[i:i+batch_size]
            ids = [d["chunk_id"] for d in batch]
            texts = [d["text"] for d in batch]
            embeddings = [d["embedding"] for d in batch]
            metadatas = [self._chunk_row_to_chroma_metadata(d) for d in batch]
            coll.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        collection_name = config.get("chromadb", {}).get("collection_name", "pdf_chunks")
        logger.info("Chroma ingest done: total_written=%d, collection=%s", len(new_table), collection_name)
        
    def search_by_text(self, query_text: str) -> dict:
        coll = self._init_chroma_client()
        client = OpenAI(
            base_url=config.get("embedding", {}).get("api_base"),
            api_key=os.getenv("QWEN_API_KEY"),
        )
        resp = client.embeddings.create(
            model=config.get("embedding", {}).get("model"),
            input=[query_text],
            encoding_format="float",
        )
        query_embedding = resp.data[0].embedding
        return coll.query(query_embeddings=[query_embedding], n_results=self.top_k, include=["documents", "metadatas", "distances"])

    def _read_all_chunk(self, all_chunk_path):
        with open(all_chunk_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    
    def _extract_text_and_metadata(self, data):
        text_list = [d["text"] for d in data]
        metadata_list = [
            {
                "file_name": d["file_name"],
                "page_index": d["page_index"],
                "chunk_id": d["chunk_id"],
                "chunk_index": d["chunk_index"],
                "char_count": d["char_count"],
            }
            for d in data
        ]
        return text_list, metadata_list

    def _batch_text(self, text_list):
        for i in range(0, len(text_list), self.batch_size):
            text_batch = text_list[i:i+self.batch_size]
            yield text_batch

    def _embed_text_batch(self, text_batch):
        client = OpenAI(
            base_url=config.get("embedding", {}).get("api_base"),
            api_key=os.getenv("QWEN_API_KEY")
        )
        response = client.embeddings.create(
            model=config.get("embedding", {}).get("model"), 
            input=text_batch,
            encoding_format="float"
        )

        return [d.embedding for d in response.data]

    def embed_chunks(self, all_chunk_data) -> list[dict]:
        text_list, _ = self._extract_text_and_metadata(all_chunk_data)
        new_table = []
        start = 0
        for text_batch in self._batch_text(text_list):
            vectors = self._embed_text_batch(text_batch)
            for i, vector in enumerate(vectors):
                idx = start + i
                chunk = all_chunk_data[idx]
                new_row = {**chunk, "embedding": vector}
                new_table.append(new_row)
            start += len(text_batch)
        return new_table
