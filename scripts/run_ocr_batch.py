import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.9")
import paddle
paddle.set_flags({"FLAGS_fraction_of_gpu_memory_to_use": 0.9})
import json
from tqdm import tqdm
from src.ocr_processor import OCRProcessor
from src.utils import load_config
import gc
import logging
logger = logging.getLogger(__name__)
config = load_config()

def process_all_pdfs():
    config = load_config()
    processor = OCRProcessor(config)
    input_dir = config.get("paths", {}).get("pdf_dir", "data/raw/OpenDataLab___OmniDocBench/pdfs")
    pdf_files = processor.list_pdf_files(directory=input_dir)

    # 过滤已有缓存的文件，跳过不重跑
    pending = [f for f in pdf_files if not processor._build_cache_path(f, "structured").exists()]
    logger.info("总文件 %d，已缓存 %d，待处理 %d", len(pdf_files), len(pdf_files) - len(pending), len(pending))

    batch_size = config.get("ocr", {}).get("paddleocrv1", {}).get("batch_size", 4)

    for i in tqdm(range(0, len(pending), batch_size), desc="batch"):
        batch = pending[i:i + batch_size]
        for pdf_path in batch:
            try:
                processor.process_pdf(pdf_path)
            except Exception as e:
                logger.exception("处理文件 %s 时发生错误", pdf_path)

        gc.collect()
        try:
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
        except Exception:
            pass

    logger.info("批量处理完成")
            
if __name__ == "__main__":
    process_all_pdfs()
