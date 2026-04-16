#!/usr/bin/env python3
"""视频批量抽帧管线：读取视频路径 -> decord/opencv 抽帧 -> 可选过滤/去重 -> 可选上传 HDFS -> 保存 meta jsonl"""

import argparse
import os
import shutil
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from data.preprocess.utils import get_hdfs_dir, upload_dir_to_hdfs, prune_img_paths
from data.preprocess.modules import PixelEntropy, PHash, Embedding, deduplicate_cc


# ==================== 配置 ====================

@dataclass
class PipelineConfig:
    input_txt: str
    output_frames_dir: str
    output_meta_dir: str
    target_fps: float = 1.0
    min_frames: int = 10
    
    # 功能开关
    enable_entropy: bool = True
    enable_phash: bool = True
    enable_sscd: bool = True
    upload_dir: Optional[str] = None  # HDFS 上传目录，None 表示不上传
    
    # 阈值参数
    entropy_threshold: float = 0.3
    phash_threshold: float = 1 - 14/64
    sscd_threshold: float = 0.75
    large_video_extra_phash: float = 2/64
    large_video_extra_sscd: float = 0.05
    large_video_threshold: int = 6000

    def __post_init__(self):
        if self.upload_dir is not None:
            self.upload_dir = get_hdfs_dir(self.upload_dir)
        self.output_frames_dir = os.path.abspath(self.output_frames_dir)


# ==================== 抽帧模块 ====================

def _get_frame_indices(total_frames: int, video_fps: float, target_fps: float) -> np.ndarray:
    if target_fps >= video_fps:
        return np.arange(total_frames)
    duration = total_frames / video_fps
    n_frames = max(1, int(duration * target_fps))
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=np.int64)
    return np.unique(indices)


def _decord_worker(video_path: str, target_fps: float, q: Queue):
    from decord import VideoReader, cpu
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
        indices = _get_frame_indices(total_frames, video_fps, target_fps)
        frames = vr.get_batch(indices).asnumpy()
        q.put(("ok", frames, indices, video_fps))
    except Exception as e:
        q.put(("error", str(e)))


def extract_with_decord(video_path: str, target_fps: float, timeout: int = 30):
    q = Queue()
    p = Process(target=_decord_worker, args=(video_path, target_fps, q))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.kill()
        p.join()
        raise RuntimeError(f"decord timeout ({timeout}s): {video_path}")

    if q.empty():
        raise RuntimeError(f"decord worker crashed: {video_path}")

    result = q.get()
    if result[0] == "error":
        raise RuntimeError(f"decord error: {result[1]}")

    _, frames, indices, video_fps = result
    return frames, indices, video_fps


def extract_with_opencv(video_path: str, target_fps: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames <= 0 or video_fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video meta: {video_path}")
    
    indices = _get_frame_indices(total_frames, video_fps, target_fps)
    frames = []
    prev_idx = -1
    
    for idx in indices:
        idx = int(idx)
        if idx - prev_idx > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame[:, :, ::-1])  # BGR -> RGB
        prev_idx = idx
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read: {video_path}")
    return np.array(frames), indices[:len(frames)], video_fps


def _save_single(args: Tuple[np.ndarray, str]) -> str:
    frame_rgb, save_path = args
    frame_bgr = frame_rgb[:, :, ::-1]
    cv2.imwrite(save_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return save_path


def save_frames(frames: np.ndarray, indices: np.ndarray, output_dir: str, workers: int = 4) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    n_digits = max(4, len(str(len(indices))))
    tasks = [
        (frame, os.path.join(output_dir, f"frame_{i:0{n_digits}d}_idx{int(idx):06d}.jpg"))
        for i, (frame, idx) in enumerate(zip(frames, indices))
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_save_single, tasks))


def extract_frames(video_path: str, target_fps: float, output_dir: str) -> Tuple[List[str], np.ndarray, float]:
    """抽帧并保存，返回 (img_paths, indices, video_fps)"""
    try:
        frames, indices, video_fps = extract_with_decord(video_path, target_fps)
    except Exception:
        frames, indices, video_fps = extract_with_opencv(video_path, target_fps)

    if len(frames) == 0:
        return [], np.array([]), 0.0

    saved_paths = save_frames(frames, indices, output_dir)
    return saved_paths, indices, video_fps


# ==================== 过滤/去重模块 ====================

class VideoProcessor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.pix_entropy = PixelEntropy(batch_size=256, num_workers=10, device="cuda") if cfg.enable_entropy else None
        self.p_hash = PHash() if cfg.enable_phash else None
        self.sscd_emb = Embedding(batch_size=256, num_workers=10, device="cuda") if cfg.enable_sscd else None

    def filter_by_entropy(self, img_paths: List[str], meta_infos: List[Dict]) -> Tuple[List[str], List[Dict]]:
        if not self.cfg.enable_entropy or len(img_paths) == 0:
            return img_paths, meta_infos
        
        scores = self.pix_entropy.encode_images(img_paths)
        filtered_paths, filtered_metas = [], []
        for path, meta, score in zip(img_paths, meta_infos, scores):
            if score >= self.cfg.entropy_threshold:
                meta["entropy"] = round(float(score), 2)
                filtered_paths.append(path)
                filtered_metas.append(meta)
        return filtered_paths, filtered_metas

    def dedupe_by_phash(self, img_paths: List[str], meta_infos: List[Dict]) -> Tuple[List[str], List[Dict]]:
        if not self.cfg.enable_phash or len(img_paths) == 0:
            return img_paths, meta_infos
        
        threshold = self.cfg.phash_threshold
        if len(img_paths) >= self.cfg.large_video_threshold:
            threshold += self.cfg.large_video_extra_phash

        vectors = self.p_hash.encode_images(img_paths)
        unique_ids = deduplicate_cc(vectors, index_type="BinaryFlat", K=256, T=threshold)
        return [img_paths[i] for i in unique_ids], [meta_infos[i] for i in unique_ids]

    def dedupe_by_sscd(self, img_paths: List[str], meta_infos: List[Dict]) -> Tuple[List[str], List[Dict]]:
        if not self.cfg.enable_sscd or len(img_paths) == 0:
            return img_paths, meta_infos
        
        threshold = self.cfg.sscd_threshold
        if len(img_paths) >= self.cfg.large_video_threshold:
            threshold += self.cfg.large_video_extra_sscd

        vectors = self.sscd_emb.encode_images(img_paths)
        unique_ids = deduplicate_cc(vectors, index_type="FlatIP", K=256, T=threshold)
        return [img_paths[i] for i in unique_ids], [meta_infos[i] for i in unique_ids]

    def process(self, img_paths: List[str], meta_infos: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """依次执行 entropy -> phash -> sscd 过滤"""
        paths, metas = self.filter_by_entropy(img_paths, meta_infos)
        paths, metas = self.dedupe_by_phash(paths, metas)
        paths, metas = self.dedupe_by_sscd(paths, metas)
        return paths, metas


# ==================== 主管线 ====================

def safe_rmtree(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def get_video_id(video_path: str) -> str:
    """从视频路径提取唯一标识（不含扩展名的文件名）"""
    return os.path.splitext(os.path.basename(video_path))[0]


def process_single_video(
    video_path: str,
    processor: VideoProcessor,
    cfg: PipelineConfig,
) -> bool:
    """处理单个视频，返回是否成功"""
    video_id = get_video_id(video_path)
    
    # 检查是否已处理
    meta_jsonl = os.path.join(cfg.output_meta_dir, f"{video_id}.jsonl")
    
    # 抽帧
    frames_dir = os.path.join(cfg.output_frames_dir, video_id)
    try:
        img_paths, indices, video_fps = extract_frames(video_path, cfg.target_fps, frames_dir)
    except Exception as e:
        print(f"[ERROR] Extract {video_path}: {e}")
        return False

    if len(img_paths) < cfg.min_frames:
        print(f"[SKIP] {video_id}: frames ({len(img_paths)}) < {cfg.min_frames}")
        safe_rmtree(frames_dir)
        return False

    # 构建 meta
    meta_infos = [
        {
            "video_id": video_id,
            "fps": video_fps,
            "img_path": path,
            "frame_idx": int(idx),
            "timestamp": round(float(idx) / video_fps, 4) if video_fps > 0 else 0,
        }
        for path, idx in zip(img_paths, indices)
    ]

    # 过滤/去重
    original_paths = list(img_paths)
    final_paths, final_metas = processor.process(img_paths, meta_infos)

    # 删除未保留的帧文件
    prune_img_paths(original_paths, final_paths)

    # 上传并清理
    if cfg.upload_dir is not None:
        try:
            upload_dir_to_hdfs(frames_dir, cfg.upload_dir)
        except Exception as e:
            print(f"[WARN] Upload failed for {video_id}: {e}")
        safe_rmtree(frames_dir)

    # 写 meta jsonl
    with open(meta_jsonl, "w", encoding="utf-8") as f:
        for meta in final_metas:
            if cfg.upload_dir is not None:
                meta["img_path"] = meta["img_path"].replace(cfg.output_frames_dir, cfg.upload_dir)
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"[INFO] {video_id} done — {len(final_paths)} frames")
    return True


def process_videos_from_txt(cfg: PipelineConfig):
    os.makedirs(cfg.output_frames_dir, exist_ok=True)
    os.makedirs(cfg.output_meta_dir, exist_ok=True)

    # 读取视频路径列表
    with open(cfg.input_txt, "r", encoding="utf-8") as f:
        video_paths = [line.strip() for line in f if line.strip()]

    # resume: 跳过已完成的视频
    done_ids = {os.path.splitext(f)[0] for f in os.listdir(cfg.output_meta_dir) if f.endswith(".jsonl")}
    pending_paths = [p for p in video_paths if get_video_id(p) not in done_ids]
    
    if len(done_ids) > 0:
        print(f"[RESUME] Skipped {len(video_paths) - len(pending_paths)} videos, {len(pending_paths)} remaining")

    processor = VideoProcessor(cfg)
    
    success_count = 0
    for video_path in tqdm(pending_paths, desc="Processing Videos"):
        if not os.path.isfile(video_path):
            print(f"[WARN] File not found: {video_path}")
            continue
        if process_single_video(video_path, processor, cfg):
            success_count += 1

    print(f"[INFO] Done: {success_count}/{len(pending_paths)} videos processed successfully.")


# ==================== CLI ====================

def parse_args():
    parser = argparse.ArgumentParser(description="Video frame extraction pipeline (per-video processing).")
    parser.add_argument("--input", required=True, help="Input TXT file (one video path per line)")
    parser.add_argument("--output_frames_dir", required=True, help="Output frames directory")
    parser.add_argument("--output_meta_dir", required=True, help="Output meta directory")
    parser.add_argument("--target_fps", type=float, default=1.0, help="Target FPS for extraction")
    parser.add_argument("--min_frames", type=int, default=10, help="Min frames per video")
    
    # 功能开关
    parser.add_argument("--no-entropy", action="store_true", help="Disable entropy filtering")
    parser.add_argument("--no-phash", action="store_true", help="Disable pHash deduplication")
    parser.add_argument("--no-sscd", action="store_true", help="Disable SSCD deduplication")
    parser.add_argument("--upload-dir", type=str, default=None, help="HDFS upload directory (default: no upload)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    cfg = PipelineConfig(
        input_txt=args.input,
        output_frames_dir=args.output_frames_dir,
        output_meta_dir=args.output_meta_dir,
        target_fps=args.target_fps,
        min_frames=args.min_frames,
        enable_entropy=not args.no_entropy,
        enable_phash=not args.no_phash,
        enable_sscd=not args.no_sscd,
        upload_dir=args.upload_dir,
    )

    process_videos_from_txt(cfg)