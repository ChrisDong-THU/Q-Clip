"""边框检测器：从 TOS 下载 zip 压缩包，检测图像边框并计算众数 bbox。"""

import argparse
import io
import json
import os
import re
import shutil
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import bytedtos
import numpy as np
from PIL import Image
from tqdm import tqdm

from dinov3.data.transforms import BorderCropper, ResizeShortEdge


# ============== TOS Client ==============

class TosClient:
    """TOS 对象存储客户端。"""

    def __init__(
        self,
        bucket_name: str,
        ak: str,
        tos_psm: str,
        tos_cluster: str,
        endpoint: str,
        delimiter: str = "/",
        max_keys: int = 1000,
        timeout: int = 30,
    ):
        self.client = bytedtos.Client(
            bucket_name, ak,
            service=tos_psm, cluster=tos_cluster, endpoint=endpoint,
            timeout=timeout, connect_timeout=timeout,
        )
        self.delimiter = delimiter
        self.max_keys = max_keys

    def list_all_objects(self, prefix: str) -> list[dict]:
        """列举目录下所有对象（自动分页）。"""
        all_objects, start_after, is_truncated = [], "", True
        while is_truncated:
            try:
                resp = self.client.list_prefix(prefix, self.delimiter, start_after, self.max_keys)
                data = resp.json["payload"]
                all_objects.extend(data["objects"])
                start_after, is_truncated = data["startAfter"], data["isTruncated"]
            except Exception as e:
                print(f"[ERROR] List objects failed: {prefix}, {e}")
                break
        return all_objects

    def download(self, key: str, local_path: str) -> bool:
        """下载对象到本地文件。"""
        try:
            resp = self.client.get_object(key)
            with open(local_path, "wb") as f:
                f.write(resp.data)
            return True
        except Exception as e:
            print(f"[ERROR] Download failed: {key}, {e}")
            return False


# ============== Bbox Detector ==============

# 匹配 <episode_id>_frames.zip
ZIP_PATTERN = re.compile(r"^(\d+)_frames\.zip$")
# 匹配 <frame_id>.jpg
FRAME_PATTERN = re.compile(r"^(\d+)\.jpg$")


def parse_zip_name(name: str) -> Optional[int]:
    """解析 zip 文件名，返回 episode_id。"""
    match = ZIP_PATTERN.match(name)
    return int(match.group(1)) if match else None


def parse_frame_name(name: str) -> Optional[int]:
    """解析帧文件名，返回 frame_id。"""
    match = FRAME_PATTERN.match(name)
    return int(match.group(1)) if match else None


class BboxDetector:
    """边框检测器，支持多线程并行处理。"""

    def __init__(self, num_workers: int = 8):
        self.num_workers = num_workers
        self._local = threading.local()

    def _get_cropper(self) -> BorderCropper:
        """每个线程独立的 BorderCropper 实例。"""
        if not hasattr(self._local, "cropper"):
            self._local.cropper = BorderCropper()
        return self._local.cropper

    def _detect_one(self, img_path: str) -> Optional[tuple[int, int, int, int]]:
        """检测单张图像的边框。"""
        try:
            img = Image.open(img_path).convert("RGB")
            return self._get_cropper().detect_bbox(img)
        except Exception as e:
            print(f"[ERROR] Detect failed: {img_path}, {e}")
            return None

    def detect_batch(self, img_paths: list[str]) -> list[Optional[tuple[int, int, int, int]]]:
        """并行检测一批图像的边框。"""
        results = [None] * len(img_paths)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._detect_one, p): i for i, p in enumerate(img_paths)}

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=60)
                except Exception as e:
                    print(f"[ERROR] Detection timeout: {img_paths[idx]}, {e}")

        return results


# ============== Album Processor ==============

def extract_sample_frames(
    tos_client: TosClient,
    tos_dir: str,
    tmp_dir: str,
    sample_ratio: float = 0.5,
) -> tuple[list[str], Optional[Image.Image]]:
    """
    下载并解压 zip，返回采样后的帧路径列表和第一张图像。
    
    Args:
        sample_ratio: 采样比例，0.5 表示每隔一帧取一帧
    """
    # 列举并过滤 zip 文件
    objects = tos_client.list_all_objects(tos_dir)
    zip_objects = []
    for obj in objects:
        name = Path(obj["key"]).name
        episode_id = parse_zip_name(name)
        if episode_id is not None:
            zip_objects.append((episode_id, obj["key"]))

    if not zip_objects:
        return [], None

    zip_objects.sort(key=lambda x: x[0])

    # 下载、解压、收集帧路径
    all_frame_paths = []
    first_img = None

    for episode_id, obj_key in zip_objects:
        zip_path = os.path.join(tmp_dir, f"{episode_id}_frames.zip")

        if not tos_client.download(obj_key, zip_path):
            continue

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # 收集有效帧并排序
                frames = []
                for name in zf.namelist():
                    frame_id = parse_frame_name(Path(name).name)
                    if frame_id is not None:
                        frames.append((frame_id, name))

                frames.sort(key=lambda x: x[0])

                # 解压帧文件
                for frame_id, name in frames:
                    local_path = os.path.join(tmp_dir, f"{episode_id}_{frame_id}.jpg")
                    with zf.open(name) as src, open(local_path, "wb") as dst:
                        dst.write(src.read())
                    all_frame_paths.append(local_path)

                    # 保存第一张图像
                    if first_img is None:
                        try:
                            first_img = Image.open(local_path).convert("RGB")
                        except Exception:
                            pass
        except Exception as e:
            print(f"[ERROR] Extract failed: {zip_path}, {e}")
        finally:
            # 删除 zip 释放空间
            if os.path.exists(zip_path):
                os.remove(zip_path)

    # 采样
    step = max(1, int(1 / sample_ratio))
    sampled_paths = all_frame_paths[::step]

    return sampled_paths, first_img


def compute_mode_bbox(
    bboxes: list[Optional[tuple[int, int, int, int]]],
    bias: int = 10,
    threshold: float = 0.5,
) -> Optional[tuple[int, int, int, int]]:
    """计算众数 bbox，若有效比例低于阈值则返回 None。"""
    valid_bboxes = [b for b in bboxes if b is not None]
    if not valid_bboxes:
        return None

    arr = np.array(valid_bboxes)
    mode_bbox = tuple(np.median(arr, axis=0).round().astype(int).tolist())

    # 检查有效比例
    valid_count = np.sum(
        (np.abs(arr[:, 0] - mode_bbox[0]) <= bias) &
        (np.abs(arr[:, 1] - mode_bbox[1]) <= bias) &
        (np.abs(arr[:, 2] - mode_bbox[2]) <= bias) &
        (np.abs(arr[:, 3] - mode_bbox[3]) <= bias)
    )

    if valid_count / len(valid_bboxes) < threshold:
        return None

    return mode_bbox


def process_album(
    tos_client: TosClient,
    detector: BboxDetector,
    tos_dir: str,
    sample_ratio: float = 0.5,
    batch_size: int = 256,
    bias: int = 10,
    threshold: float = 0.5,
) -> Optional[dict]:
    """处理单个 album，返回 bbox 和 resize_size 信息。"""
    tmp_dir = tempfile.mkdtemp(prefix="bbox_detect_")

    try:
        # 下载并解压
        frame_paths, first_img = extract_sample_frames(tos_client, tos_dir, tmp_dir, sample_ratio)

        if not frame_paths or first_img is None:
            print(f"[WARN] No valid frames: {tos_dir}")
            return None

        # 批量检测 bbox
        all_bboxes = []
        for i in range(0, len(frame_paths), batch_size):
            batch = frame_paths[i : i + batch_size]
            bboxes = detector.detect_batch(batch)
            all_bboxes.extend(bboxes)

        # 计算众数 bbox
        mode_bbox = compute_mode_bbox(all_bboxes, bias, threshold)

        # 计算 resize_size
        if mode_bbox:
            x1, y1, x2, y2 = mode_bbox
            w, h = x2 - x1, y2 - y1
        else:
            w, h = first_img.size

        resize_size = ResizeShortEdge().detect_size((w, h))

        return {
            "tos_dir": tos_dir,
            "bbox": mode_bbox,
            "resize_size": resize_size,
        }

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============== Utils ==============

def load_task_list(path: str) -> list[str]:
    """加载任务列表文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ============== Main ==============

def process_albums(
    tos_config: dict,
    task_txt: str,
    sample_ratio: float = 0.5,
    batch_size: int = 256,
    bias: int = 10,
    threshold: float = 0.5,
    num_workers: int = 16,
) -> None:
    """批量处理所有 albums，输出 bbox 信息到 JSONL。"""
    tos_dirs = load_task_list(task_txt)
    print(f"[INFO] Processing {len(tos_dirs)} albums")

    tos_client = TosClient(**tos_config)
    detector = BboxDetector(num_workers)

    output_path = task_txt.replace(".txt", ".jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for tos_dir in tqdm(tos_dirs, desc="Processing albums"):
            result = process_album(
                tos_client, detector, tos_dir,
                sample_ratio, batch_size, bias, threshold,
            )

            if result:
                print(f"[INFO] {tos_dir}: bbox={result['bbox']}, resize_size={result['resize_size']}")
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()


def main():
    parser = argparse.ArgumentParser(
        description="边框检测器（ZIP 模式）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task-txt", required=True, help="TOS 目录列表文件")
    parser.add_argument("--sample-ratio", type=float, default=0.125, help="帧采样比例")
    parser.add_argument("--batch-size", type=int, default=256, help="批处理大小")
    parser.add_argument("--bias", type=int, default=10, help="bbox 容差")
    parser.add_argument("--threshold", type=float, default=0.5, help="有效 bbox 比例阈值")
    parser.add_argument("--num-workers", type=int, default=16, help="并行线程数")
    args = parser.parse_args()

    tos_config = {
        "ak": "8X4STUMNH9EIFO5FFHE1",
        "bucket_name": "toutiao-aigc-video-frames",
        "tos_psm": "toutiao.tos.tosapi",
        "tos_cluster": "default",
        "endpoint": "tos-cn-north.byted.org",
        "delimiter": "/",
        "max_keys": 100,
    }

    process_albums(
        tos_config,
        args.task_txt,
        args.sample_ratio,
        args.batch_size,
        args.bias,
        args.threshold,
        args.num_workers,
    )


if __name__ == "__main__":
    main()
