"""JSONL 文件列举工具：列举目录下所有 JSONL 文件并输出到文件。"""

import argparse
from pathlib import Path


def list_jsonl_files(jsonl_dir: str | Path, output_txt: str | Path) -> int:
    """
    列举目录下所有 JSONL 文件并输出到文件。
    """
    jsonl_dir = Path(jsonl_dir)
    output_txt = Path(output_txt)

    # 获取所有 JSONL 文件
    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"[ERROR] No JSONL files found in {jsonl_dir}")
        return 0

    # 确保输出目录存在
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    # 写入输出文件
    with open(output_txt, "w", encoding="utf-8") as f:
        for path in jsonl_files:
            f.write(f"{path.absolute()}\n")

    print(f"[DONE] {len(jsonl_files)} files written to {output_txt}")
    return len(jsonl_files)


if __name__ == "__main__":
    list_jsonl_files("/mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/ai_image_recognition_v3/20260113/dedup", "tmp_tasks_hdfs_0113/short_drama_dedup/task_all.txt")
