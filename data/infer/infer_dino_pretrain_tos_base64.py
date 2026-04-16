"""DINOv3 TOS 特征提取器：从 TOS 存储批量加载图像并提取特征。"""

import argparse
import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Optional

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
    """TOS 对象存储客户端，支持目录列举和对象下载。"""

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
            bucket_name,
            ak,
            service=tos_psm,
            cluster=tos_cluster,
            endpoint=endpoint,
            timeout=timeout,
            connect_timeout=timeout,
        )
        self.delimiter = delimiter
        self.max_keys = max_keys

    def list_objects(self, prefix: str, start_after: str = "") -> tuple[list[dict], str, bool]:
        """列举目录下所有对象，返回 (对象列表, 下次游标, 是否截断)。"""
        try:
            resp = self.client.list_prefix(prefix, self.delimiter, start_after, self.max_keys)
            data = resp.json["payload"]
            return data["objects"], data["startAfter"], data["isTruncated"]
        except Exception as e:
            print(f"[ERROR] List objects failed: {prefix}, {e}")
            return [], "", False

    def list_all_objects(self, prefix: str) -> list[dict]:
        """列举目录下所有对象（自动分页）。"""
        all_objects, start_after, is_truncated = [], "", True
        while is_truncated:
            objects, start_after, is_truncated = self.list_objects(prefix, start_after)
            all_objects.extend(objects)
        return all_objects

    def download(self, key: str) -> Optional[bytes]:
        """下载单个对象，返回原始字节。"""
        try:
            resp = self.client.get_object(key)
            if len(resp.data) < 200:
                print(f"[WARN] Skip bad object (too small): {key}")
                return None
            return resp.data
        except Exception as e:
            print(f"[ERROR] Download failed: {key}, {e}")
            return None


# ============== Dataset ==============

class TosImageDataset(Dataset):
    """从 TOS 加载图像的 Dataset，支持 DataLoader 多进程。"""

    def __init__(
        self,
        objects: list[dict],
        tos_config: dict[str, Any],
        crop_bbox: Optional[tuple[int, int, int, int]] = None,
        resize_size: tuple[int, int] = (224, 224),
    ):
        self.objects = objects
        self.tos_config = tos_config
        self._client: Optional[TosClient] = None

        self.transform = v2.Compose([
            v2.Lambda(lambda img: img.crop(crop_bbox) if crop_bbox else img),
            v2.ToImage(),
            v2.Resize((resize_size[1], resize_size[0]), interpolation=v2.InterpolationMode.BICUBIC),
            make_base_transform(),
        ])

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, idx: int) -> tuple[int, Optional[dict], Optional[torch.Tensor]]:
        obj_key = self.objects[idx]["key"]
        result = self._fetch_and_process(obj_key)

        if result is None:
            return idx, None, None

        frame_info, img = result
        return idx, frame_info, self.transform(img)

    def _get_client(self) -> TosClient:
        """延迟初始化客户端（支持多进程）。"""
        if self._client is None:
            self._client = TosClient(**self.tos_config)
        return self._client

    def _fetch_and_process(self, key: str) -> Optional[tuple[dict, Image.Image]]:
        """下载并解析 TOS 对象，返回 (frame_info, image)。"""
        raw_data = self._get_client().download(key)
        if raw_data is None:
            return None

        # 解析 JSON
        try:
            record = json.loads(raw_data)
        except json.JSONDecodeError:
            print(f"[ERROR] JSON decode failed: {key}")
            return None

        frame_info = {"frame_id": record["frame_id"], "entity_key": record["entity_key"]}

        # 解码 base64 图像
        img_base64 = record.get("img_data")
        if not img_base64:
            print(f"[WARN] No img_data field: {key}")
            return None

        # 处理可能的 data:image/xxx;base64, 前缀
        if "," in img_base64[:60]:
            img_base64 = img_base64.split(",", 1)[1]

        try:
            jpg_bytes = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Image decode failed: {key}, {e}")
            return None

        return frame_info, img


def collate_skip_none(batch: list) -> tuple[list[int], list[dict], torch.Tensor]:
    """过滤加载失败的样本，返回 (失败索引, 有效帧信息, 有效图像张量)。"""
    bad_indices, valid_infos, valid_images = [], [], []

    for idx, frame_info, img in batch:
        if img is None:
            bad_indices.append(idx)
        else:
            valid_infos.append(frame_info)
            valid_images.append(img)

    images_tensor = torch.stack(valid_images) if valid_images else torch.empty(0)
    return bad_indices, valid_infos, images_tensor


# ============== Model ==============

def build_default_model() -> tuple[nn.Module, torch.dtype]:
    """加载默认的 DINOv3 预训练模型。"""
    torch.backends.cudnn.benchmark = True

    work_dir = Path("logs/pretrain_vitg16_toutiao/train-1")
    cfg_path = work_dir / "config.yaml"
    ckpt_path = work_dir / "eval/training_249999/teacher_checkpoint/teacher_checkpoint.pth"

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
        batch_size: int = 128,
        num_workers: int = 8,
    ):
        self.model = model
        self.autocast_dtype = autocast_dtype
        self.batch_size = batch_size
        self.num_workers = num_workers

    def extract(
        self,
        objects: list[dict],
        tos_config: dict[str, Any],
        crop_bbox: Optional[tuple[int, int, int, int]] = None,
        resize_size: tuple[int, int] = (224, 224),
    ) -> tuple[np.ndarray, list[int], list[dict]]:
        """提取特征，返回 (特征矩阵, 失败索引, 帧信息列表)。"""
        dataset = TosImageDataset(objects, tos_config, crop_bbox, resize_size)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_skip_none,
        )

        all_features, all_bad_indices, all_frame_infos = [], [], []

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
            for bad_indices, frame_infos, images in tqdm(loader, desc="Extracting", leave=False):
                all_bad_indices.extend(bad_indices)
                all_frame_infos.extend(frame_infos)

                if images.numel() == 0:
                    continue

                features = self.model(images.cuda())
                all_features.append(features.cpu().numpy())

        result = np.concatenate(all_features, axis=0) if all_features else np.empty((0,))
        return result, all_bad_indices, all_frame_infos


# ============== Utils ==============

def load_jsonl(path: str | Path) -> list[dict]:
    """加载 JSONL 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_video_name(tos_dir: str) -> str:
    """从 TOS 目录路径提取视频名称。"""
    return tos_dir.rstrip("/").split("/")[-1]


# ============== Main ==============

def process_videos(
    tos_dir_infos: list[dict],
    tos_config: dict[str, Any],
    output_dir: str,
    batch_size: int = 128,
    num_workers: int = 8,
) -> None:
    """批量处理视频，提取特征并保存。"""
    print(f"[INFO] Processing {len(tos_dir_infos)} videos")

    os.makedirs(output_dir, exist_ok=True)
    tos_client = TosClient(**tos_config)
    model, autocast_dtype = build_default_model()
    extractor = DINOv3Extractor(model, autocast_dtype, batch_size, num_workers)

    for info in tqdm(tos_dir_infos, desc="Processing videos"):
        name = get_video_name(info["tos_dir"])
        objects = tos_client.list_all_objects(info["tos_dir"])

        features, bad_indices, frame_infos = extractor.extract(
            objects,
            tos_config,
            info.get("bbox"),
            info.get("resize_size", (224, 224)),
        )

        # 过滤失败样本并校验
        bad_set = set(bad_indices)
        valid_infos = [fi for i, fi in enumerate(frame_infos) if i not in bad_set]
        assert len(features) == len(valid_infos), \
            f"Mismatch: {len(features)} features vs {len(valid_infos)} frame_infos"

        # 保存特征和元数据
        np.save(os.path.join(output_dir, f"{name}.npy"), features)

        with open(os.path.join(output_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for fi in valid_infos:
                f.write(json.dumps(fi, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="DINOv3 TOS 批量特征提取",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task-jsonl", required=True, help="任务列表 JSONL 路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
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
    process_videos(tos_dir_infos, tos_config, args.output_dir, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()