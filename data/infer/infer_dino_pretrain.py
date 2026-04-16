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
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dinov3.data.transforms import make_base_transform
from dinov3.eval.setup import build_model_for_eval, get_autocast_dtype


# ============== Dataset ==============

class ImageDataset(Dataset):
    """支持 HDFS 路径的图像数据集，自动完成裁剪、缩放和归一化。"""

    def __init__(
        self,
        paths: list[str],
        crop_bbox: Optional[tuple[int, int, int, int]] = None,
        resize_size: tuple[int, int] = (224, 224),
    ):
        self.paths = paths
        self.transform = v2.Compose([
            v2.Lambda(lambda img: img.crop(crop_bbox) if crop_bbox else img),
            v2.ToImage(),
            v2.Resize((resize_size[1], resize_size[0]), interpolation=v2.InterpolationMode.BICUBIC),
            make_base_transform(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[int, Optional[torch.Tensor]]:
        path = self.paths[idx]
        try:
            with fsspec.open(path, "rb") as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
            return idx, self.transform(img)
        except Exception:
            print(f"[ERROR] Failed to load: {path}")
            return idx, None


def collate_skip_none(batch: list) -> tuple[list[int], torch.Tensor]:
    """过滤加载失败的样本，返回 (失败索引列表, 有效图像张量)。"""
    bad_indices, valid_images = [], []
    for idx, img in batch:
        if img is None:
            bad_indices.append(idx)
        else:
            valid_images.append(img)
    
    images_tensor = torch.stack(valid_images) if valid_images else torch.empty(0)
    return bad_indices, images_tensor


# ============== Model ==============

def build_default_model() -> tuple[nn.Module, torch.dtype]:
    """加载默认的 DINOv3 预训练模型。"""
    torch.backends.cudnn.benchmark = True

    work_dir = Path("dinov3/pretrained/short_drama") # movie
    cfg_path = work_dir / "config.yaml"
    ckpt_path = work_dir / "teacher_checkpoint.pth"

    model_cfg = OmegaConf.load(cfg_path)
    autocast_dtype = get_autocast_dtype(model_cfg)
    model = build_model_for_eval(model_cfg, ckpt_path)

    return model.cuda(), autocast_dtype


# ============== Inference ==============

class DINOv3Extractor:
    """DINOv3 特征提取器，封装模型加载与批量推理。"""

    def __init__(
        self,
        model: nn.Module,
        autocast_dtype: torch.dtype,
        batch_size: int = 256,
        num_workers: int = 8,
    ):
        self.model = model
        self.autocast_dtype = autocast_dtype
        self.batch_size = batch_size
        self.num_workers = num_workers

    def extract(
        self,
        paths: list[str],
        crop_bbox: Optional[tuple[int, int, int, int]] = None,
        resize_size: tuple[int, int] = (224, 224),
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

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
            for bad_indices, batch in loader:
                all_bad_indices.extend(bad_indices)
                
                if batch.numel() == 0:
                    continue
                
                features = self.model(batch.cuda())
                all_features.append(features.cpu().numpy())

        result = np.concatenate(all_features, axis=0) if all_features else np.empty((0,))
        return result, all_bad_indices


# ============== Utils ==============

def load_jsonl(path: str | Path) -> list[dict]:
    """加载 JSONL 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ============== Main ==============

def process_tasks(
    jsonl_infos: list[dict],
    output_dir: str,
    batch_size: int = 256,
    num_workers: int = 8,
):
    """批量处理任务列表，提取特征并保存。"""
    os.makedirs(output_dir, exist_ok=True)
    
    model, autocast_dtype = build_default_model()
    extractor = DINOv3Extractor(model, autocast_dtype, batch_size, num_workers)

    for task in tqdm(jsonl_infos, desc="Processing tasks"):
        jsonl_path = task["jsonl_path"]
        records = load_jsonl(jsonl_path)
        img_paths = [r["img_path"] for r in records]

        features, bad_indices = extractor.extract(
            img_paths,
            task.get("bbox"),
            task.get("resize_size", (224, 224)),
        )

        # 过滤失败样本并校验
        valid_records = [r for i, r in enumerate(records) if i not in set(bad_indices)]
        assert len(features) == len(valid_records), \
            f"Mismatch: {len(features)} features vs {len(valid_records)} records"

        # 保存特征
        name = Path(jsonl_path).stem
        np.save(os.path.join(output_dir, f"{name}.npy"), features)


def main():
    parser = argparse.ArgumentParser(description="DINOv3 批量特征提取")
    parser.add_argument("--task-jsonl", required=True, help="任务列表 JSONL 路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    tasks = load_jsonl(args.task_jsonl)
    process_tasks(tasks, args.output_dir, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()