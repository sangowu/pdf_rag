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

    logger.info("开始批量处理 %d 个文件...", len(pdf_files))

    for filename in tqdm(pdf_files):
        fcount = 0
        pdf_path = filename
        try:
            processor.process_pdf(pdf_path)
            fcount += 1
        except Exception as e:
            logger.error("处理文件 %s 时发生错误: %s", filename, e)
        if fcount % 5 == 0:
            import gc
            gc.collect()
            try:
                if paddle.device.is_compiled_with_cuda():
                    paddle.device.cuda.empty_cache()
            except Exception:
                pass

    logger.info("批量处理完成")
            
if __name__ == "__main__":
    process_all_pdfs()
