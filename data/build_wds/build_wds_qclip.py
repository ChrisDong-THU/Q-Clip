"""Embedding 分片构建器：从预计算的 npy 文件切分 clips 并打包为 WebDataset 分片。"""

import argparse
import io
import json
import os
import random
import tarfile
import tempfile
import time
import uuid
from contextlib import suppress
from pathlib import Path

import fsspec
import numpy as np
from tqdm import tqdm


# ============== Utils ==============

def load_jsonl(path: str | Path) -> list[dict]:
    """加载 JSONL 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_task_list(path: str | Path) -> list[str]:
    """加载任务列表文件（每行一个 jsonl 路径）。"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def traverse_clips(embeddings: np.ndarray, meta_infos: list[dict], clip_len: int = 512) -> list[tuple[np.ndarray, list[dict]]]:
    """
    以 50% 重叠滑动窗口切分 embeddings 和 meta_infos 为 clips。
    
    若总帧数 <= clip_len，返回空列表（不切分）。
    """
    n, _ = embeddings.shape
    if n <= clip_len:
        return []

    half = clip_len // 2
    starts = np.arange(0, n, half)
    starts = starts[starts < n - clip_len]
    starts = np.append(starts, n - clip_len)  # 确保最后一个 clip 完整

    return [(embeddings[s : s + clip_len], meta_infos[s : s + clip_len]) for s in starts]


# ============== Shard Writer ==============

class ShardWriter:
    """WebDataset 分片写入器，支持 HDFS 上传和重试。"""

    def __init__(
        self,
        output_dir: str,
        prefix: str,
        shard_size: int = 100,
        upload_retries: int = 5,
        backoff_cap: float = 60.0,
    ):
        self.fs = fsspec.filesystem("hdfs")
        self.output_dir = output_dir.rstrip("/")
        self.prefix = prefix
        self.shard_size = shard_size
        self.upload_retries = upload_retries
        self.backoff_cap = backoff_cap

        self.shard_idx = 0
        self.sample_idx = 0
        self._buffer: list[dict[str, bytes]] = []

    def add_sample(self, embeddings: np.ndarray, meta_infos: list[dict], album_id: str, pair_id: int) -> None:
        """添加一个样本到缓冲区，满时自动刷新。"""
        sample_id = f"{self.prefix}_{self.shard_idx:06d}_{self.sample_idx:06d}"
        self._buffer.append({
            "key": sample_id,
            "npy": embeddings.tobytes(),
            "jsonl": "\n".join(json.dumps(m, ensure_ascii=False) for m in meta_infos).encode("utf-8"),
            "json": json.dumps(
                {"album_id": album_id, "pair_id": pair_id},
                ensure_ascii=False,
            ).encode("utf-8"),
        })

        self.sample_idx += 1
        if self.sample_idx >= self.shard_size:
            self._flush()

    def finalize(self) -> None:
        """刷新剩余缓冲区。"""
        if self._buffer:
            self._flush()

    def _flush(self) -> None:
        """将缓冲区写入 tar 并上传到 HDFS。"""
        hdfs_path = f"{self.output_dir}/{self.prefix}_{self.shard_idx:06d}.tar"

        local_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f"wds_{self.prefix}_{self.shard_idx:06d}_",
                suffix=".tar",
                delete=False,
            ) as tmp:
                local_path = tmp.name
                with tarfile.open(fileobj=tmp, mode="w") as tar:
                    for sample in self._buffer:
                        key = sample["key"]
                        self._add_file(tar, f"{key}.npy", sample["npy"])
                        self._add_file(tar, f"{key}.jsonl", sample["jsonl"])
                        self._add_file(tar, f"{key}.json", sample["json"])

            self._upload_with_retry(local_path, hdfs_path)
        finally:
            if local_path and os.path.exists(local_path):
                with suppress(OSError):
                    os.remove(local_path)

        self._buffer.clear()
        self.shard_idx += 1
        self.sample_idx = 0

    def _upload_with_retry(self, local_path: str, hdfs_path: str) -> None:
        """带重试的 HDFS 上传。"""
        upload_id = uuid.uuid4().hex

        for attempt in range(self.upload_retries + 1):
            tmp_hdfs_path = f"{hdfs_path}.tmp.{upload_id}.{attempt}"
            try:
                self.fs.put(local_path, tmp_hdfs_path)

                with suppress(Exception):
                    self.fs.rm(hdfs_path)
                self.fs.mv(tmp_hdfs_path, hdfs_path)
                return

            except Exception as e:
                with suppress(Exception):
                    self.fs.rm(tmp_hdfs_path)

                if attempt >= self.upload_retries:
                    raise RuntimeError(
                        f"Upload failed after {self.upload_retries + 1} attempts: {hdfs_path}"
                    ) from e

                backoff = min(self.backoff_cap, 2**attempt) + random.random()
                time.sleep(backoff)

    @staticmethod
    def _add_file(tar: tarfile.TarFile, name: str, data: bytes) -> None:
        """向 tar 添加文件。"""
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


# ============== Main ==============

def build_shards(
    task_txt: str,
    emb_dir: str,
    output_dir: str,
    prefix: str,
    shard_size: int = 100,
    clip_len: int = 512,
) -> None:
    """从预计算的 npy 文件构建 WebDataset 分片。"""
    jsonl_paths = load_task_list(task_txt)
    print(f"[INFO] Building shards: {len(jsonl_paths)} tasks, emb={emb_dir} -> {output_dir}")

    writer = ShardWriter(output_dir, prefix, shard_size)
    skipped = 0

    for jsonl_path in tqdm(jsonl_paths, desc="Building shards"):
        album_id = Path(jsonl_path).stem
        npy_path = os.path.join(emb_dir, f"{album_id}.npy")

        if not os.path.isfile(npy_path):
            skipped += 1
            continue

        if not os.path.isfile(jsonl_path):
            skipped += 1
            continue

        embeddings = np.load(npy_path)
        meta_infos = load_jsonl(jsonl_path)

        if len(embeddings) != len(meta_infos):
            print(f"[WARN] Length mismatch for {album_id}: emb={len(embeddings)}, info={len(meta_infos)}")
            skipped += 1
            continue

        clips = traverse_clips(embeddings, meta_infos, clip_len)

        for pair_id, (clip_emb, clip_meta) in enumerate(clips):
            writer.add_sample(clip_emb, clip_meta, album_id, pair_id)

    writer.finalize()
    print(f"[DONE] Created {writer.shard_idx} shards ({skipped} albums skipped)")


def main():
    parser = argparse.ArgumentParser(
        description="Embedding 分片构建器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task-txt", required=True, help="任务列表文件路径")
    parser.add_argument("--emb-dir", required=True, help="npy 文件目录")
    parser.add_argument("--output-dir", required=True, help="输出 HDFS 目录")
    parser.add_argument("--output-prefix", required=True, help="分片文件前缀")
    parser.add_argument("--shard-size", type=int, default=400, help="每个分片的样本数")
    parser.add_argument("--clip-len", type=int, default=512, help="每个 clip 的帧数")
    args = parser.parse_args()

    build_shards(
        args.task_txt,
        args.emb_dir,
        args.output_dir,
        args.output_prefix,
        args.shard_size,
        args.clip_len,
    )


if __name__ == "__main__":
    main()