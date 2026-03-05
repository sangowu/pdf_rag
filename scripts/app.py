import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.8"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import tempfile
import uvicorn
import gc
from pathlib import Path
from paddleocr import PPStructureV3
import paddle

app = FastAPI(title="PaddleOCR REST API")

try:
    paddle.set_flags({'FLAGS_fraction_of_gpu_memory_to_use': 0.8})
except:
    pass

print("正在加载 PPStructureV3 模型（GPU）...")
pipeline = PPStructureV3(device="gpu", precision="fp16")
print("模型加载完成")

def warm_up():
    print("正在执行模型预热（强制载入 GPU 显存）...")
    try:
        fake_img = np.zeros((512, 512, 3), dtype=np.uint8)
        pipeline.predict(input=fake_img)
        print("预热完成，所有子模型已就绪。")
    except Exception as e:
        print(f"预热失败，可能显存不足: {e}")

warm_up()

RESULTS_DIR = Path("/root/autodl-tmp/results/ocr_outputs")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 全局计数器用于触发显存清理
processed_count = 0

@app.post("/ocr_batch")
async def ocr_batch(files: list[UploadFile] = File(...)):
    global processed_count
    results_status = []

    for file in files:
        original_name = file.filename
        original_stem = Path(original_name).stem
        
        # 1. 写入临时文件
        tmp_path = Path(tempfile.gettempdir()) / original_name
        with open(tmp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            # 2. 运行预测
            results = pipeline.predict(input=str(tmp_path))

            # 3. 保存结果
            for res in results:
                res.save_to_json(save_path=str(RESULTS_DIR))
            
            # 手动释放 results 引用，协助垃圾回收
            del results

            # 4. 改进匹配逻辑：使用字符串过滤，解决特殊字符 [ ] # 导致的 glob 失败
            generated_files = [
                f.name for f in RESULTS_DIR.iterdir()
                if f.is_file() 
                and f.name.startswith(original_stem) 
                and f.name.endswith("_res.json")
            ]

            results_status.append({
                "original_filename": original_name,
                "status": "completed",
                "generated_json_files": generated_files, 
                "download_urls": [
                    f"http://127.0.0.1:8899/download/{fname}" for fname in generated_files
                ]
            })

            # 5. 每处理 5 个文件释放一次显存
            processed_count += 1
            if processed_count % 5 == 0:
                print(f"--- 已处理 {processed_count} 个文件，正在回收显存 ---")
                gc.collect() # 清理 Python 层内存
                if paddle.device.is_compiled_with_cuda():
                    paddle.device.cuda.empty_cache() # 强制归还显存池给显卡驱动
                print("--- 显存回收完成 ---")

        except Exception as e:
            results_status.append({
                "original_filename": original_name,
                "status": "error",
                "message": str(e)
            })

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return {
        "status": "batch_processed",
        "details": results_status
    }

@app.get("/download/{filename}")
async def download_file(filename: str):
    safe_filename = Path(filename).name 
    file_path = RESULTS_DIR / safe_filename
    
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="JSON结果文件未找到")
    
    return FileResponse(
        path=file_path,
        filename=safe_filename,
        media_type="application/json"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8899, workers=1)