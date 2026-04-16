"""MP4 文件列举工具：列举目录下所有 MP4 文件并输出到文件。"""

import argparse
from pathlib import Path


def list_mp4_files(mp4_dir: str | Path, output_txt: str | Path) -> int:
    """
    列举目录下所有 MP4 文件并输出到文件。
    """
    mp4_dir = Path(mp4_dir)
    output_txt = Path(output_txt)

    # 获取所有 MP4 文件
    mp4_files = sorted(mp4_dir.glob("*.mp4"))

    if not mp4_files:
        print(f"[ERROR] No MP4 files found in {mp4_dir}")
        return 0

    # 确保输出目录存在
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    # 写入输出文件
    with open(output_txt, "w", encoding="utf-8") as f:
        for path in mp4_files:
            f.write(f"{path.absolute()}\n")

    print(f"[DONE] {len(mp4_files)} files written to {output_txt}")
    return len(mp4_files)


if __name__ == "__main__":
    list_mp4_files("/mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/ai_image_recognition_v3/kinetics-dataset/k400/train/", "tmp_k400_tasks/train/task_all.txt")