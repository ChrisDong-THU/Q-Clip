"""DINOv3 特征提取器：从 JSONL 任务列表批量提取图像特征并保存。"""

import argparse
import io
import json
import os
from pathlib import Path
from typing import Optional

import fsspec
import numpy as np
import torch
import torchvision.transforms.v2 as v2
from modelscope import pipeline # pip install modelscope[framework]
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============== Dataset ==============

class ImageDataset(Dataset):
    """支持 HDFS 路径的图像数据集，自动完成裁剪和缩放。"""

    def __init__(
        self,
        paths: list[str],
        crop_bbox: Optional[tuple[int, int, int, int]] = None,
        resize_size: tuple[int, int] = (256, 256),
    ):
        self.paths = paths
        self.transform = v2.Compose([
            v2.Lambda(lambda img: img.crop(crop_bbox) if crop_bbox else img),
            v2.Resize((resize_size[1], resize_size[0]), interpolation=v2.InterpolationMode.BICUBIC),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[int, Optional[Image.Image]]:
        path = self.paths[idx]
        try:
            with fsspec.open(path, "rb") as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
            return idx, self.transform(img)
        except Exception:
            print(f"[ERROR] Failed to load: {path}")
            return idx, None


def collate_skip_none(batch: list) -> tuple[list[int], list[Image.Image]]:
    """过滤加载失败的样本，返回 (失败索引列表, 有效图像列表)。"""
    bad_indices, valid_images = [], []
    for idx, img in batch:
        (bad_indices if img is None else valid_images).append(idx if img is None else img)
        if img is not None:
            bad_indices  # 占位，实际逻辑已在上行
    # 重写更清晰
    bad_indices, valid_images = [], []
    for idx, img in batch:
        if img is None:
            bad_indices.append(idx)
        else:
            valid_images.append(img)
    return bad_indices, valid_images


# ============== Inference ==============

class DINOv3Extractor:
    """DINOv3 特征提取器，封装模型加载与批量推理。"""

    def __init__(self, model_path: str = "dinov3/pretrained/release_vitb16", batch_size: int = 32, num_workers: int = 4):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pipeline = pipeline(
            model=model_path,
            task="image-feature-extraction",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def extract(
        self,
        paths: list[str],
        crop_bbox: Optional[tuple[int, int, int, int]] = None,
        resize_size: tuple[int, int] = (256, 256),
    ) -> tuple[np.ndarray, list[int]]:
        """提取特征，返回 (特征矩阵, 失败索引列表)。"""
        dataset = ImageDataset(paths, crop_bbox, resize_size)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_skip_none,
        )

        all_features, all_bad_indices = [], []
        for bad_indices, images in tqdm(loader, desc="Extracting", leave=False):
            all_bad_indices.extend(bad_indices)
            if not images:
                continue

            for feat in self.pipeline(images):
                feat = np.asarray(feat)
                feat = feat[0, 0]  # CLS token
                all_features.append(feat)

        features = np.stack(all_features) if all_features else np.empty((0,))
        return features, all_bad_indices


# ============== Utils ==============

def load_jsonl(path: str | Path) -> list[dict]:
    """加载 JSONL 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ============== Main ==============

def process_tasks(jsonl_infos: list[dict], output_dir: str, batch_size: int = 32, num_workers: int = 10):
    """批量处理任务列表，提取特征并保存。"""
    os.makedirs(output_dir, exist_ok=True)
    extractor = DINOv3Extractor(batch_size=batch_size, num_workers=num_workers)

    for task in tqdm(jsonl_infos, desc="Processing tasks"):
        jsonl_path = task["jsonl_path"]
        records = load_jsonl(jsonl_path)
        img_paths = [r["img_path"] for r in records]

        features, bad_indices = extractor.extract(img_paths, task.get("bbox"), task.get("resize_size", (256, 256)))

        # 过滤失败样本
        valid_records = [r for i, r in enumerate(records) if i not in set(bad_indices)]
        assert len(features) == len(valid_records), f"Mismatch: {len(features)} features vs {len(valid_records)} records"

        # 保存
        name = Path(jsonl_path).stem
        np.save(os.path.join(output_dir, f"{name}.npy"), features)


def main():
    parser = argparse.ArgumentParser(description="DINOv3 批量特征提取")
    parser.add_argument("--task-jsonl", required=True, help="任务列表 JSONL 路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=10)
    args = parser.parse_args()

    tasks = load_jsonl(args.task_jsonl)
    process_tasks(tasks, args.output_dir, args.batch_size, args.num_workers)


if __name__ == "__main__":
    # main()

    process_tasks(
        load_jsonl("tmp_task_debug.jsonl"), 
        "tmp_debug", 
        32, 
        10
    )