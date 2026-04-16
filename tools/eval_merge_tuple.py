"""Embedding 合并工具：将多个 npy + jsonl 文件对合并为分片输出。"""

import argparse
import json
from glob import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm


def save_shard(
    features: list[np.ndarray],
    labels: list[dict],
    output_emb_dir: Path,
    output_info_dir: Path,
    prefix: str,
    shard_idx: int,
) -> None:
    """保存单个分片（npy 和 jsonl 分别输出到不同目录）。"""
    filename = f"{prefix}_{shard_idx:06d}"

    np.save(output_emb_dir / f"{filename}.npy", np.concatenate(features, axis=0))

    with open(output_info_dir / f"{filename}.jsonl", "w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + "\n")


def random_sample(
    features: np.ndarray,
    labels: list[dict],
    ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict]]:
    """随机采样指定比例的数据。"""
    n = len(labels)
    k = max(1, int(n * ratio))
    indices = rng.choice(n, size=k, replace=False)
    indices.sort()

    return features[indices], [labels[i] for i in indices]


def merge_embeddings(
    input_emb_dir: str | Path,
    input_info_dir: str | Path,
    output_emb_dir: str | Path,
    output_info_dir: str | Path,
    prefix: str,
    shard_size: int = 500000,
    preserve_ratio: float = 1.0,
    seed: int = 42,
) -> int:
    """
    合并目录下所有 npy + jsonl 文件对为分片输出。
    
    Args:
        input_emb_dir: 输入 npy 文件目录
        input_info_dir: 输入 jsonl 文件目录
        output_emb_dir: 输出 npy 文件目录
        output_info_dir: 输出 jsonl 文件目录
        prefix: 输出文件前缀
        shard_size: 每个分片的最大样本数
        preserve_ratio: 每个子文件随机保留的比例 (0, 1]
        seed: 随机种子
    
    Returns:
        生成的分片数量
    """
    input_emb_dir = Path(input_emb_dir)
    input_info_dir = Path(input_info_dir)
    output_emb_dir = Path(output_emb_dir)
    output_info_dir = Path(output_info_dir)

    # 查找所有 jsonl 文件
    jsonl_files = sorted(glob(str(input_info_dir / "*.jsonl")))

    if not jsonl_files:
        print(f"[ERROR] No JSONL files found in {input_info_dir}")
        return 0

    print(f"[INFO] Found {len(jsonl_files)} files in {input_info_dir}")
    if preserve_ratio < 1.0:
        print(f"[INFO] Preserve ratio: {preserve_ratio:.2%}")

    output_emb_dir.mkdir(parents=True, exist_ok=True)
    output_info_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 合并文件
    shard_idx = 0  # XXX: 可以改为其他初始值，避免覆盖已有文件
    curr_labels: list[dict] = []
    curr_features: list[np.ndarray] = []

    for jsonl_path in tqdm(jsonl_files, desc="Merging"):
        jsonl_path = Path(jsonl_path)
        npy_path = input_emb_dir / jsonl_path.with_suffix(".npy").name

        if not npy_path.exists():
            print(f"[WARN] Missing npy: {npy_path}")
            continue

        # 加载数据
        with open(jsonl_path, "r", encoding="utf-8") as f:
            labels = [json.loads(line) for line in f if line.strip()]

        features = np.load(npy_path)

        if features.shape[0] != len(labels):
            print(f"[WARN] Shape mismatch: {npy_path} ({features.shape[0]}) vs {jsonl_path} ({len(labels)})")
            continue

        # 随机采样
        if preserve_ratio < 1.0:
            features, labels = random_sample(features, labels, preserve_ratio, rng)

        curr_labels.extend(labels)
        curr_features.append(features)

        # 达到分片大小则保存
        if len(curr_labels) >= shard_size:
            save_shard(curr_features, curr_labels, output_emb_dir, output_info_dir, prefix, shard_idx)
            shard_idx += 1
            curr_labels.clear()
            curr_features.clear()

    # 保存剩余数据
    if curr_labels:
        save_shard(curr_features, curr_labels, output_emb_dir, output_info_dir, prefix, shard_idx)
        shard_idx += 1

    print(f"[DONE] Created {shard_idx} shards")
    print(f"       Embeddings: {output_emb_dir}")
    print(f"       Info: {output_info_dir}")
    return shard_idx


def main():
    parser = argparse.ArgumentParser(
        description="Embedding 合并工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-emb-dir", required=True, help="输入 npy 文件目录")
    parser.add_argument("--input-info-dir", required=True, help="输入 jsonl 文件目录")
    parser.add_argument("--output-emb-dir", required=True, help="输出 npy 文件目录")
    parser.add_argument("--output-info-dir", required=True, help="输出 jsonl 文件目录")
    parser.add_argument("--prefix", "-p", required=True, help="输出文件前缀")
    parser.add_argument("--shard-size", type=int, default=500000, help="每个分片的最大样本数")
    parser.add_argument("--preserve-ratio", type=float, default=1.0, help="每个子文件随机保留的比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    merge_embeddings(
        args.input_emb_dir,
        args.input_info_dir,
        args.output_emb_dir,
        args.output_info_dir,
        args.prefix,
        args.shard_size,
        args.preserve_ratio,
        args.seed,
    )


if __name__ == "__main__":
    main()