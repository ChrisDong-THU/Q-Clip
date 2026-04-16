"""边框检测器：从 JSONL 文件列表批量检测图像边框并计算众数 bbox。"""

import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, Optional

import fsspec
import numpy as np
from PIL import Image
from tqdm import tqdm

from dinov3.data.transforms import BorderCropper, ResizeShortEdge


# ============== Utils ==============

def load_jsonl(path: str | Path) -> list[dict]:
    """加载 JSONL 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_task_list(path: str | Path) -> list[str]:
    """加载任务列表文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ============== Bbox Detector ==============

class BboxDetector:
    """边框检测器，支持多线程并行处理。"""

    def __init__(self, num_workers: int = 16):
        self.num_workers = num_workers
        self.border_cropper = BorderCropper()

    def _fetch_image(self, path: str) -> Optional[Image.Image]:
        """从 HDFS 或本地读取图像。"""
        try:
            with fsspec.open(path, "rb") as f:
                return Image.open(io.BytesIO(f.read())).convert("RGB")
        except Exception:
            return None

    def _detect_one(self, path: str) -> Optional[tuple[int, int, int, int]]:
        """检测单张图像的边框。"""
        img = self._fetch_image(path)
        if img is None:
            return None
        return self.border_cropper.detect_bbox(img)

    def detect_batch(self, paths: list[str]) -> Iterator[tuple[str, Optional[tuple[int, int, int, int]]]]:
        """并行检测一批图像的边框。"""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._detect_one, p): p for p in paths}

            for future in as_completed(futures):
                path = futures[future]
                try:
                    bbox = future.result(timeout=30)
                    yield path, bbox
                except Exception as e:
                    print(f"[ERROR] Failed: {path}, {e}")
                    yield path, None


# ============== Bbox Computation ==============

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
    detector: BboxDetector,
    img_paths: list[str],
    batch_size: int = 256,
    bias: int = 10,
    threshold: float = 0.5,
) -> tuple[Optional[tuple[int, int, int, int]], tuple[int, int]]:
    """
    处理单个 album，返回 (mode_bbox, resize_size)。
    """
    # 获取第一张图像用于计算默认尺寸
    first_img = detector._fetch_image(img_paths[0])

    # 批量检测 bbox
    all_bboxes = []
    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i : i + batch_size]
        for _, bbox in detector.detect_batch(batch):
            all_bboxes.append(bbox)

    # 计算众数 bbox
    mode_bbox = compute_mode_bbox(all_bboxes, bias, threshold)

    # 计算 resize_size
    if mode_bbox:
        x1, y1, x2, y2 = mode_bbox
        w, h = x2 - x1, y2 - y1
    else:
        w, h = first_img.size if first_img else (224, 224)

    resize_size = ResizeShortEdge().detect_size((w, h))

    return mode_bbox, resize_size


# ============== Main ==============

def process_jsonl_list(
    task_txt: str,
    sample_ratio: float = 0.5,
    batch_size: int = 256,
    bias: int = 10,
    threshold: float = 0.5,
    num_workers: int = 16,
) -> None:
    """批量处理 JSONL 文件列表，输出 bbox 信息。"""
    jsonl_files = load_task_list(task_txt)
    print(f"[INFO] Processing {len(jsonl_files)} JSONL files")

    detector = BboxDetector(num_workers)
    output_path = task_txt.replace(".txt", "_bbox.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for jsonl_path in tqdm(jsonl_files, desc="Processing"):
            records = load_jsonl(jsonl_path)
            img_paths = [r["img_path"] for r in records]

            if not img_paths:
                print(f"[WARN] Empty: {jsonl_path}")
                continue

            # 采样
            step = max(1, int(1 / sample_ratio))
            sampled_paths = img_paths[::step]

            try:
                mode_bbox, resize_size = process_album(
                    detector, sampled_paths, batch_size, bias, threshold
                )
            except Exception as e:
                print(f"[ERROR] {jsonl_path}: {e}")
                continue

            print(f"[INFO] {jsonl_path}: bbox={mode_bbox}, resize_size={resize_size}")

            result = {
                "jsonl_path": jsonl_path,
                "bbox": mode_bbox,
                "resize_size": resize_size,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    print(f"[DONE] Output written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="边框检测器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task-txt", required=True, help="JSONL 文件列表")
    parser.add_argument("--sample-ratio", type=float, default=0.5, help="帧采样比例")
    parser.add_argument("--batch-size", type=int, default=256, help="批处理大小")
    parser.add_argument("--bias", type=int, default=10, help="bbox 容差")
    parser.add_argument("--threshold", type=float, default=0.5, help="有效 bbox 比例阈值")
    parser.add_argument("--num-workers", type=int, default=16, help="并行线程数")
    args = parser.parse_args()

    process_jsonl_list(
        args.task_txt,
        args.sample_ratio,
        args.batch_size,
        args.bias,
        args.threshold,
        args.num_workers,
    )


if __name__ == "__main__":
    main()