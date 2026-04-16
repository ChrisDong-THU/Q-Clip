"""文件分片工具：将 JSONL 或 TXT 文件均匀拆分为多个分片。"""

import argparse
from pathlib import Path


def split_file(input_path: str | Path, num_shards: int, output_dir: str | Path) -> list[str]:
    """
    将文件均匀拆分为 num_shards 个分片。
    
    Args:
        input_path: 输入文件路径（支持 .jsonl 或 .txt）
        num_shards: 分片数量
        output_dir: 输出目录
    
    Returns:
        输出文件路径列表
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 校验文件类型
    suffix = input_path.suffix.lower()
    if suffix not in (".jsonl", ".txt"):
        raise ValueError(f"Unsupported file type '{suffix}', must be .jsonl or .txt")

    # 读取所有非空行
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    total = len(lines)
    print(f"[INFO] Read {total} lines from {input_path}")

    if total == 0:
        print("[ERROR] Input file is empty")
        return []

    # 调整分片数
    if num_shards > total:
        print(f"[WARN] num_shards ({num_shards}) > total ({total}), adjusting to {total}")
        num_shards = total

    # 计算分片大小
    base_size = total // num_shards
    remainder = total % num_shards

    # 生成分片文件
    output_paths = []
    offset = 0

    for i in range(num_shards):
        # 前 remainder 个分片多分一行
        chunk_size = base_size + (1 if i < remainder else 0)
        chunk = lines[offset : offset + chunk_size]
        offset += chunk_size

        output_path = output_dir / f"task_node_{i}{suffix}"

        with open(output_path, "w", encoding="utf-8") as f:
            for line in chunk:
                f.write(line if line.endswith("\n") else line + "\n")

        output_paths.append(str(output_path))
        print(f"  [shard {i}] {output_path.name}: {len(chunk)} lines")

    print(f"[DONE] Split into {num_shards} files in {output_dir}")
    return output_paths


if __name__ == "__main__":
    split_file("tmp_tasks_hdfs_0113/short_drama_dedup/task_all.txt", 64, "tmp_tasks_hdfs_0113/short_drama_dedup/")
