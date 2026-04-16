"""向量检索工具：使用 FAISS 进行大规模向量检索"""

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from tqdm import tqdm


# ============== Utils ==============

def load_jsonl(path: str | Path) -> list[dict]:
    """加载 JSONL 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2 归一化。"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def load_shards(shard_paths: list[str], normalize: bool = True) -> tuple[np.ndarray, list[dict]]:
    """加载并合并分片数据，可选 L2 归一化。"""
    all_vectors, all_records = [], []

    for path in tqdm(shard_paths, desc="Loading shards"):
        vectors = np.load(path).astype(np.float32)
        records = load_jsonl(Path(path).with_suffix(".jsonl"))

        if normalize:
            vectors = l2_normalize(vectors)

        all_vectors.append(vectors)
        all_records.extend(records)

    return np.concatenate(all_vectors, axis=0), all_records


# ============== Retrieval ==============

def build_index(
    dim: int,
    train_vectors: np.ndarray,
    pca_dim: Optional[int] = None,
    use_gpu: bool = True,
) -> faiss.Index:
    """
    构建 FAISS 索引，支持 PCA Whitening 预处理。
    """
    # 使用 index_factory 构建索引
    # PCAW = PCA + Whitening, Flat = 精确内积搜索
    if pca_dim:
        factory_string = f"PCAW{pca_dim},Flat"
    else:
        factory_string = "Flat"

    index = faiss.index_factory(dim, factory_string, faiss.METRIC_INNER_PRODUCT)

    # 训练索引（PCA 需要训练）
    if pca_dim:
        print(f"[INFO] Training index with factory string: {factory_string}")
        index.train(train_vectors)

    if use_gpu and faiss.get_num_gpus() > 0:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = False
        index = faiss.index_cpu_to_all_gpus(index, co)
        print(f"[INFO] Index moved to {faiss.get_num_gpus()} GPUs")

    return index


def search_batch(
    index: faiss.Index,
    query_vectors: np.ndarray,
    top_k: int = 10,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """批量检索。"""
    all_distances, all_indices = [], []

    for i in tqdm(range(0, len(query_vectors), batch_size), desc="Searching"):
        batch = query_vectors[i : i + batch_size]
        distances, indices = index.search(batch, top_k)
        all_distances.append(distances)
        all_indices.append(indices)

    return np.concatenate(all_distances), np.concatenate(all_indices)


def save_results(
    output_path: Path,
    query_records: list[dict],
    index_records: list[dict],
    distances: np.ndarray,
    indices: np.ndarray,
) -> int:
    """保存检索结果。"""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for q_record, dists, idxs in zip(query_records, distances, indices):
            for dist, idx in zip(dists, idxs):
                if idx < 0:
                    continue
                result = {
                    "q_record": q_record,
                    "tgt_record": index_records[idx],
                    "score": float(dist),
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1
    return count


# ============== Main ==============

def run_retrieval(
    work_dir: str,
    top_k: int = 10,
    batch_size: int = 4096,
    pca_dim: Optional[int] = None,
) -> None:
    """执行向量检索。"""
    work_dir = Path(work_dir)
    result_dir = work_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    # 收集分片路径
    index_dir = work_dir / "index_merged"
    query_dir = work_dir / "query_merged"

    index_shards = sorted(glob(str(index_dir / "index_*.npy")))
    query_shards = sorted(glob(str(query_dir / "query_*.npy")))

    print(f"[INFO] Index shards: {len(index_shards)}, Query shards: {len(query_shards)}")

    # 加载数据（L2 归一化）
    index_vectors, index_records = load_shards(index_shards, normalize=True)
    query_vectors, query_records = load_shards(query_shards, normalize=True)

    print(f"[INFO] Index: {len(index_records)}, Query: {len(query_records)}")

    # 构建索引
    dim = index_vectors.shape[1]
    index = build_index(dim, index_vectors, pca_dim, use_gpu=True)

    # 添加向量到索引
    index.add(index_vectors)

    # 检索
    distances, indices = search_batch(index, query_vectors, top_k, batch_size)

    # 保存结果
    pca_suffix = f"_pca_{pca_dim}" if pca_dim else "_no_pca"
    output_path = result_dir / f"retrieval_results{pca_suffix}.jsonl"
    count = save_results(output_path, query_records, index_records, distances, indices)

    print(f"[DONE] Saved {count} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="向量检索工具（支持 L2 归一化和 PCA Whitening）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--work-dir", required=True, help="工作目录")
    parser.add_argument("--top-k", type=int, default=10, help="检索 top-k")
    parser.add_argument("--batch-size", type=int, default=4096, help="批处理大小")
    parser.add_argument("--pca-dim", type=int, default=None, help="PCA Whitening 目标维度（不指定则不使用）")
    args = parser.parse_args()

    run_retrieval(args.work_dir, args.top_k, args.batch_size, args.pca_dim)


if __name__ == "__main__":
    main()