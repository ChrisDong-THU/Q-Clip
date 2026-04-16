"""DINOv3 TOS 特征提取器：从 TOS 下载 zip 压缩包，解压提取帧图像并批量推理。"""

import argparse
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import bytedtos
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


# ============== Dataset ==============

class LocalImageDataset(Dataset):
    """从本地路径加载图像的 Dataset。"""

    def __init__(
        self,
        frame_infos: list[dict],
        crop_bbox: Optional[tuple[int, int, int, int]] = None,
        resize_size: tuple[int, int] = (224, 224),
    ):
        self.frame_infos = frame_infos
        self.transform = v2.Compose([
            v2.Lambda(lambda img: img.crop(crop_bbox) if crop_bbox else img),
            v2.ToImage(),
            v2.Resize((resize_size[1], resize_size[0]), interpolation=v2.InterpolationMode.BICUBIC),
            make_base_transform(),
        ])

    def __len__(self) -> int:
        return len(self.frame_infos)

    def __getitem__(self, idx: int) -> tuple[int, Optional[torch.Tensor]]:
        info = self.frame_infos[idx]
        try:
            img = Image.open(info["local_path"]).convert("RGB")
            return idx, self.transform(img)
        except Exception as e:
            print(f"[ERROR] Failed to load: {info['local_path']}, {e}")
            return idx, None


def collate_skip_none(batch: list) -> tuple[list[int], torch.Tensor]:
    """过滤加载失败的样本。"""
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
    """DINOv3 特征提取器。"""

    def __init__(
        self,
        model: nn.Module,
        autocast_dtype: torch.dtype,
        batch_size: int = 128,
        num_workers: int = 8,
    ):
        self.model = model
        self.autocast_dtype = autocast_dtype
        self.batch_size = batch_size
        self.num_workers = num_workers

    def extract(
        self,
        frame_infos: list[dict],
        crop_bbox: Optional[tuple[int, int, int, int]] = None,
        resize_size: tuple[int, int] = (224, 224),
    ) -> tuple[np.ndarray, list[int]]:
        """提取特征，返回 (特征矩阵, 失败索引列表)。"""
        dataset = LocalImageDataset(frame_infos, crop_bbox, resize_size)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_skip_none,
        )

        all_features, all_bad_indices = [], []

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
            for bad_indices, images in loader:
                all_bad_indices.extend(bad_indices)
                if images.numel() == 0:
                    continue
                features = self.model(images.cuda())
                all_features.append(features.cpu().numpy())

        result = np.concatenate(all_features, axis=0) if all_features else np.empty((0,))
        return result, all_bad_indices


# ============== Album Processor ==============

# 匹配 <episode_id>_frames.zip
ZIP_PATTERN = re.compile(r"^(\d+)_frames\.zip$")
# 匹配 <frame_id>.jpg
FRAME_PATTERN = re.compile(r"^(\d+)\.jpg$")


def parse_zip_name(name: str) -> Optional[int]:
    """解析 zip 文件名，返回 episode_id，不匹配返回 None。"""
    match = ZIP_PATTERN.match(name)
    return int(match.group(1)) if match else None


def parse_frame_name(name: str) -> Optional[int]:
    """解析帧文件名，返回 frame_id，不匹配返回 None。"""
    match = FRAME_PATTERN.match(name)
    return int(match.group(1)) if match else None


def extract_frames_from_zip(zip_path: str, extract_dir: str, album_id: str, episode_id: int) -> list[dict]:
    """解压 zip 并返回有序的帧信息列表。"""
    frame_infos = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            frame_id = parse_frame_name(Path(name).name)
            if frame_id is None:
                continue

            local_path = os.path.join(extract_dir, f"{episode_id}_{frame_id}.jpg")
            with zf.open(name) as src, open(local_path, "wb") as dst:
                dst.write(src.read())

            frame_infos.append({
                "local_path": local_path,
                "frame_id": frame_id,
                "episode_id": episode_id,
                "album_id": album_id,
            })

    # 按 frame_id 排序
    frame_infos.sort(key=lambda x: x["frame_id"])
    return frame_infos


def process_album(
    tos_client: TosClient,
    extractor: DINOv3Extractor,
    tos_dir: str,
    album_id: str,
    output_emb_dir: str,
    output_info_dir: str,
    crop_bbox: Optional[tuple[int, int, int, int]] = None,
    resize_size: tuple[int, int] = (224, 224),
) -> None:
    """处理单个 album：下载 -> 解压 -> 推理 -> 保存 -> 清理。"""
    # 创建临时目录
    tmp_dir = tempfile.mkdtemp(prefix=f"album_{album_id}_")

    try:
        # 列举并过滤 zip 文件
        objects = tos_client.list_all_objects(tos_dir)
        zip_objects = []
        for obj in objects:
            name = Path(obj["key"]).name
            episode_id = parse_zip_name(name)
            if episode_id is not None:
                zip_objects.append((episode_id, obj["key"]))

        if not zip_objects:
            print(f"[WARN] No valid zip files in {tos_dir}")
            return

        # 按 episode_id 排序
        zip_objects.sort(key=lambda x: x[0])

        # 下载、解压、收集帧信息
        all_frame_infos = []
        for episode_id, obj_key in zip_objects:
            zip_path = os.path.join(tmp_dir, f"{episode_id}_frames.zip")

            if not tos_client.download(obj_key, zip_path):
                continue

            frame_infos = extract_frames_from_zip(zip_path, tmp_dir, album_id, episode_id)
            all_frame_infos.extend(frame_infos)

            # 删除 zip 释放空间
            os.remove(zip_path)

        if not all_frame_infos:
            print(f"[WARN] No valid frames in {tos_dir}")
            return

        # 推理
        features, bad_indices = extractor.extract(all_frame_infos, crop_bbox, resize_size)

        # 过滤失败样本
        bad_set = set(bad_indices)
        valid_infos = [
            {"frame_id": info["frame_id"], "episode_id": info["episode_id"], "album_id": info["album_id"]}
            for i, info in enumerate(all_frame_infos) if i not in bad_set
        ]

        assert len(features) == len(valid_infos), \
            f"Mismatch: {len(features)} features vs {len(valid_infos)} infos"

        # 保存
        np.save(os.path.join(output_emb_dir, f"{album_id}.npy"), features)

        with open(os.path.join(output_info_dir, f"{album_id}.jsonl"), "w", encoding="utf-8") as f:
            for info in valid_infos:
                f.write(json.dumps(info, ensure_ascii=False) + "\n")

    finally:
        # 清理临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============== Utils ==============

def load_jsonl(path: str | Path) -> list[dict]:
    """加载 JSONL 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_album_id(tos_dir: str) -> str:
    """从 TOS 目录路径提取 album_id。"""
    return tos_dir.rstrip("/").split("/")[-1]


# ============== Main ==============

def process_albums(
    tos_dir_infos: list[dict],
    tos_config: dict,
    output_dir: str,
    batch_size: int = 128,
    num_workers: int = 8,
) -> None:
    """批量处理所有 albums。"""
    print(f"[INFO] Processing {len(tos_dir_infos)} albums")

    output_emb_dir = os.path.join(output_dir, "emb")
    output_info_dir = os.path.join(output_dir, "emb_info")
    os.makedirs(output_emb_dir, exist_ok=True)
    os.makedirs(output_info_dir, exist_ok=True)

    tos_client = TosClient(**tos_config)
    model, autocast_dtype = build_default_model()
    extractor = DINOv3Extractor(model, autocast_dtype, batch_size, num_workers)

    for info in tqdm(tos_dir_infos, desc="Processing albums"):
        album_id = get_album_id(info["tos_dir"])

        process_album(
            tos_client, extractor,
            tos_dir=info["tos_dir"],
            album_id=album_id,
            output_emb_dir=output_emb_dir,
            output_info_dir=output_info_dir,
            crop_bbox=info.get("bbox"),
            resize_size=info.get("resize_size", (224, 224)),
        )


def main():
    parser = argparse.ArgumentParser(
        description="DINOv3 TOS 批量特征提取（ZIP 模式）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task-jsonl", required=True, help="任务列表 JSONL 路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=10)
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

    tos_dir_infos = load_jsonl(args.task_jsonl)
    process_albums(tos_dir_infos, tos_config, args.output_dir, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()