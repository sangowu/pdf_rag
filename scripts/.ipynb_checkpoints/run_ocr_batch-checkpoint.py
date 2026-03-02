import os
import json
from tqdm import tqdm
from src.ocr_processor import OCRProcessor
from src.utils import load_config

def process_all_pdfs():
    config = load_config()
    processor = OCRProcessor(config)
    input_dir = config.get("paths", {}).get("pdf_dir", "data/raw/OpenDataLab___OmniDocBench/pdfs")
    output_dir = config.get("paths", {}).get("ocr_output_dir", "results/ocr_outputs")
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]

    print(f"开始批量处理 {len(pdf_files)} 个文件...")

    for filename in tqdm(pdf_files):
        pdf_path = os.path.join(input_dir, filename)
        # output_path = os.path.join(output_dir, filename.replace('.pdf', '_ocr.json'))
        # if os.path.isfile(output_path):
        #     continue
        try:
            processor.process_pdf(pdf_path)
        except Exception as e:
            print(f"处理文件 {filename} 时发生错误: {e}")
            
if __name__ == "__main__":
    process_all_pdfs()