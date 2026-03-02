import json
import os

from tqdm import tqdm

from src.utils import load_config
config = load_config()

paths = config.get("paths", {})
GT_PATH = paths.get("omnidoc_json", "data/raw/OpenDataLab___OmniDocBench/OmniDocBench.json")

def load_omnidocbench(json_path: str) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {json_path}")
    return data

def analyze_omnidocbench(data: list[dict]) -> dict:
    if not data:
        return {}
    normalized_data = []
    for item in tqdm(data, desc="Parsing pages"):
        layout_dets = item.get("layout_dets", [])
        filtered_layout_dets = [det for det in layout_dets if det["category_type"] != "abandon"]
        page_dets = []
        for det in filtered_layout_dets:
            category_type = det["category_type"]
            order = det.get("order", 0)
            anno_id = det.get("anno_id", 0)
            text = det.get("text", "")
            page_dets.append({
                "category_type": category_type,
                "order": order,
                "anno_id": anno_id,
                "text": text
            })
        relation = item.get("extra", {}).get("relation", [])
        page_no =  item.get("page_info", {}).get("page_no", 0)
        image_path = item.get("page_info", {}).get("image_path", "")
        normalized_data.append({
            "page_dets": page_dets,
            "relation": relation,
            "page_no": page_no,
            "image_path": image_path
        })
    
    with open("data/answers/normalized_data.json", "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(normalized_data)} items to data/answers/normalized_data.json")

if __name__ == "__main__":
    json_path = GT_PATH
    omnidocbench_data = load_omnidocbench(json_path)
    analyze_omnidocbench(omnidocbench_data)