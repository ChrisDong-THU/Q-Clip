"""WebDataset 构建器：从 JSONL 文件列表批量加载图像并打包为 WebDataset 分片。"""

import argparse
import io
import json
import os
import random
import tarfile
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import Iterator, Optional

import fsspec


# ============== Utils ==============

def load_jsonl(path: str | Path) -> list[dict]:
    """加载 JSONL 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_task_list(path: str | Path) -> list[str]:
    """加载任务列表文件，每行一个 JSONL 路径。"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ============== Image Loader ==============

class ImageLoader:
    """并行图像加载器，支持 HDFS 路径。"""

    def __init__(self, num_workers: int = 8):
        self.num_workers = num_workers

    def load_batch(self, records: list[dict]) -> Iterator[tuple[dict, Optional[bytes]]]:
        """并行加载一批图像，返回 (record, image_data) 迭代器。"""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._fetch_image, rec["img_path"]): rec
                for rec in records
            }

            for future in as_completed(futures):
                record = futures[future]
                try:
                    img_data = future.result(timeout=30)
                    yield record, img_data
                except Exception as e:
                    print(f"[WARN] Failed to load {record['img_path']}: {e}")
                    yield record, None

    @staticmethod
    def _fetch_image(path: str) -> Optional[bytes]:
        """从 HDFS 获取单张图像。"""
        try:
            with fsspec.open(path, "rb") as f:
                return f.read()
        except Exception:
            return None


# ============== Shard Writer ==============

class ShardWriter:
    """WebDataset 分片写入器，支持 HDFS 上传和重试。"""

    def __init__(
        self,
        output_dir: str,
        prefix: str,
        shard_size: int = 4000,
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

    def add_sample(self, record: dict, img_data: bytes) -> None:
        """添加一个样本到缓冲区，满时自动刷新。"""
        sample_id = f"{self.prefix}_{self.shard_idx:06d}_{self.sample_idx:06d}"
        self._buffer.append({
            "key": sample_id,
            "jpg": img_data,
            "json": self._build_metadata(record),
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
            # 写入本地临时文件
            with tempfile.NamedTemporaryFile(
                prefix=f"wds_{self.prefix}_{self.shard_idx:06d}_",
                suffix=".tar",
                delete=False,
            ) as tmp:
                local_path = tmp.name
                with tarfile.open(fileobj=tmp, mode="w") as tar:
                    for sample in self._buffer:
                        key = sample["key"]
                        self._add_file(tar, f"{key}.jpg", sample["jpg"])
                        self._add_file(tar, f"{key}.json", sample["json"])

            # 上传到 HDFS
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

    @staticmethod
    def _build_metadata(record: dict) -> bytes:
        """构建样本元数据。"""
        return json.dumps({
            "gid": record["gid"],
            "album_id": record.get("album_id"),
            "chapter": record.get("chapter"),
            "frame": record.get("frame"),
        }, ensure_ascii=False).encode("utf-8")


# ============== Main ==============

def build_webdataset(
    task_list_path: str,
    output_dir: str,
    prefix: str,
    shard_size: int = 4000,
    num_workers: int = 16,
    batch_size: int = 256,
) -> None:
    """从 JSONL 文件列表构建 WebDataset。"""
    jsonl_files = load_task_list(task_list_path)
    print(f"[INFO] Processing {len(jsonl_files)} JSONL files -> {output_dir}")

    loader = ImageLoader(num_workers)
    writer = ShardWriter(output_dir, prefix, shard_size)

    for jsonl_path in jsonl_files:
        records = load_jsonl(jsonl_path)

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]

            for record, img_data in loader.load_batch(batch):
                if img_data:
                    writer.add_sample(record, img_data)

    writer.finalize()
    print(f"[DONE] Created {writer.shard_idx + 1} shards")


def main():
    parser = argparse.ArgumentParser(
        description="WebDataset 构建器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task-txt", required=True, help="任务列表文件路径")
    parser.add_argument("--output-dir", required=True, help="输出 HDFS 目录")
    parser.add_argument("--output-prefix", required=True, help="分片文件前缀<NODE_ID>")
    parser.add_argument("--shard-size", type=int, default=4000, help="每个分片的样本数")
    parser.add_argument("--num-workers", type=int, default=16, help="并行加载线程数")
    parser.add_argument("--batch-size", type=int, default=256, help="批处理大小")
    args = parser.parse_args()

    build_webdataset(
        task_list_path=args.task_txt,
        output_dir=args.output_dir,
        prefix=args.output_prefix,
        shard_size=args.shard_size,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()