"""文件合并工具：将多个分片文件按自然顺序合并为单个文件。"""

import argparse
import re
from glob import glob
from pathlib import Path


def natural_sort_key(path: str) -> list:
    """自然排序键：task_node_0, 1, 2, ..., 10（而非 0, 1, 10, 2）。"""
    parts = re.split(r"(\d+)", str(path))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def merge_files(pattern: str, output_path: str | Path) -> int:
    """
    合并匹配模式的所有文件。
    """
    # 查找并排序
    file_paths = sorted(glob(pattern), key=natural_sort_key)

    if not file_paths:
        print(f"[ERROR] No files found: {pattern}")
        return 0

    print(f"[INFO] Found {len(file_paths)} files matching '{pattern}'")

    # 准备输出目录
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 合并文件
    total_lines = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for file_path in file_paths:
            file_lines = 0

            with open(file_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    out_f.write(line if line.endswith("\n") else line + "\n")
                    file_lines += 1

            print(f"  {Path(file_path).name}: {file_lines} lines")
            total_lines += file_lines

    print(f"[DONE] Merged into {output_path}: {total_lines} lines")
    return total_lines


if __name__ == "__main__":
    merge_files("tmp_tos_tasks_0413/short_drama/task_node_*.txt", "tmp_tos_tasks_0413/short_drama/task_all.txt")
