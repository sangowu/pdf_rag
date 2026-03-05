import json
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
from src.vector_store import VectorStore

def run_embedding():
    vs = VectorStore()
    all_chunk_data = vs._read_all_chunk(vs.all_chunk_path)
    new_table = vs.embed_chunks(all_chunk_data)
    vs.add_chunks_to_chroma(new_table)

    logger.info("run_embedding done: embedded_and_ingested=%d", len(new_table))

if __name__ == "__main__":
    run_embedding()