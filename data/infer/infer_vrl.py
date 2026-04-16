#!/usr/bin/env python3
"""VRL 特征提取器：从帧级嵌入提取视频级特征并保存。"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from vrl.utils import build_vrl_model_for_eval


# ============== Utils ==============

def load_jsonl(path: str | Path) -> list[dict]:
    """加载 JSONL 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_clips(embeddings: torch.Tensor, clip_len: int = 256) -> torch.Tensor:
    """将帧嵌入切分为固定长度的 clips，返回 (num_clips, clip_len, D)。"""
    n, _ = embeddings.shape

    if n <= clip_len:
        return embeddings.unsqueeze(0)

    starts = np.arange(0, n, clip_len)
    starts[-1] = max(0, n - clip_len)  # 确保最后一个 clip 完整
    clips = [embeddings[s : s + clip_len] for s in starts]

    return torch.stack(clips)


# ============== Inference ==============

class VRLExtractor:
    """VRL 特征提取器，将帧级嵌入聚合为视频级特征。"""

    def __init__(
        self,
        model: nn.Module,
        model_dtype: torch.dtype,
        clip_len: int = 256,
        batch_size: int = 32,
    ):
        self.model = model
        self.model_dtype = model_dtype
        self.clip_len = clip_len
        self.batch_size = batch_size

    def _extract_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """对单个 batch 提取特征。"""
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=self.model_dtype):
            features = self.model(batch)
        return features

    def extract(self, frame_embeddings: torch.Tensor) -> tuple[np.ndarray, list[int]]:
        """
        提取特征，返回 (特征矩阵, clip_ids)。
        
        - clip_ids: 每个特征所属的 clip 索引
        """
        clips = sample_clips(frame_embeddings, self.clip_len)
        num_clips = clips.shape[0]

        all_features, all_clip_ids = [], []

        for i in range(0, num_clips, self.batch_size):
            batch = clips[i : i + self.batch_size]
            features = self._extract_batch(batch)
            features = F.normalize(features, dim=-1)

            b = features.shape[0]
            clip_ids = list(range(i, i + b))

            all_features.append(features)
            all_clip_ids.extend(clip_ids)

        return torch.cat(all_features).cpu().numpy(), all_clip_ids


# ============== Main ==============

def get_video_name(video_info: dict) -> str:
    """从 video_info 中提取视频名称。"""
    if "jsonl_path" in video_info:
        return Path(video_info["jsonl_path"]).stem
    return video_info["tos_dir"].rstrip("/").split("/")[-1]


def process_videos(
    ckpt_dir: str,
    video_infos: list[dict],
    frame_emb_dir: str,
    output_dir: str,
    clip_len: int = 256,
    batch_size: int = 256,
):
    """批量处理视频，提取特征并保存。"""
    print(f"[INFO] Processing {len(video_infos)} videos")

    os.makedirs(output_dir, exist_ok=True)

    ckpt_dir = Path(ckpt_dir)
    model, model_dtype = build_vrl_model_for_eval(
        ckpt_dir / "config.yaml",
        ckpt_dir / "step=20000.ckpt",
    )
    extractor = VRLExtractor(model, model_dtype, clip_len, batch_size)

    for video_info in tqdm(video_infos, desc="Processing videos"):
        name = get_video_name(video_info)

        # 加载帧嵌入
        frame_emb_path = os.path.join(frame_emb_dir, f"{name}.npy")
        frame_embeddings = torch.from_numpy(np.load(frame_emb_path)).float().cuda()

        # 提取特征
        features, clip_ids = extractor.extract(frame_embeddings)

        # 保存特征
        np.save(os.path.join(output_dir, f"{name}.npy"), features)

        # 保存索引映射
        with open(os.path.join(output_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for clip_id in clip_ids:
                record = {"entity_id": name, "clip_id": clip_id}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="VRL 批量特征提取",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task-jsonl", required=True, help="任务列表 JSONL 路径")
    parser.add_argument("--frame-emb-dir", required=True, help="帧嵌入目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--ckpt-dir", required=True, help="模型权重目录")
    parser.add_argument("--clip-len", type=int, default=256, help="每个 clip 的帧数")
    parser.add_argument("--batch-size", type=int, default=256, help="推理 batch 大小")
    args = parser.parse_args()

    video_infos = load_jsonl(args.task_jsonl)
    print(f"[INFO] Loaded {len(video_infos)} videos from {args.task_jsonl}")

    process_videos(
        args.ckpt_dir,
        video_infos,
        args.frame_emb_dir,
        args.output_dir,
        args.clip_len,
        args.batch_size,
    )


if __name__ == "__main__":
    main()