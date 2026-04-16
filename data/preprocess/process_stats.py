#!/usr/bin/env python3
"""单视频抽帧调试工具：抽帧 -> entropy+edge 过滤 -> phash 去重 -> sscd 去重，记录各操作耗时和帧数"""

import os
import shutil
import time
import cv2
import json
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
import faiss
import networkit as nk

from data.preprocess.modules import PixelEntropy, PHash, Embedding


@dataclass
class StageStats:
    stages: List[Dict] = field(default_factory=list)

    def record(self, stage: str, operation: str, elapsed: float, frames_remaining: int):
        self.stages.append({
            "stage": stage,
            "operation": operation,
            "time_sec": round(elapsed, 3),
            "frames_remaining": frames_remaining,
        })

    def print_summary(self):
        print(f"\n{'='*75}")
        print(f"{'Stage':<20} {'Operation':<25} {'Time(s)':<12} {'Frames':<10}")
        print(f"{'-'*75}")
        for s in self.stages:
            print(f"{s['stage']:<20} {s['operation']:<25} {s['time_sec']:<12.3f} {s['frames_remaining']:<10}")
        print(f"{'='*75}")

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stages, f, ensure_ascii=False, indent=2)


# ==================== 去重工具（拆分计时） ====================

def deduplicate_cc_with_stats(
    vectors: np.ndarray,
    index_type: str,
    K: int,
    T: float,
    stats: StageStats,
    stage_name: str,
    method_name: str,
):
    """去重并分别记录 Index 构建/搜索 和 CC 发现的耗时"""
    N, D = vectors.shape

    # Index + Search
    t0 = time.time()
    if index_type == "BinaryFlat":
        index = faiss.IndexBinaryFlat(D * 8)
    else:
        index = faiss.IndexFlatIP(D)
    index.add(vectors)

    if index_type == "FlatIP":
        index = faiss.index_cpu_to_all_gpus(index)

    distances, indices = index.search(vectors, K + 1)
    distances, indices = distances[:, 1:], indices[:, 1:]
    stats.record(stage_name, f"{method_name} Index", time.time() - t0, N)

    # CC Discovery
    t0 = time.time()
    row_ids = np.arange(N)[:, None]
    if index_type == "BinaryFlat":
        distances = 1 - distances / (D * 8)

    mask = (distances >= T) & (row_ids < indices)
    src, dst = np.where(mask)
    edge_indices = indices[src, dst]

    G = nk.Graph(n=N, weighted=False, directed=False)
    for u, v in zip(src, edge_indices):
        G.addEdge(int(u), int(v))

    cc = nk.components.ConnectedComponents(G).run()
    unique_ids = [comp[0] for comp in cc.getComponents()]
    stats.record(stage_name, f"{method_name} CC Discovery", time.time() - t0, len(unique_ids))

    return unique_ids


# ==================== 抽帧 ====================

def extract_frames(video_cap, fps, frame_per_second, output_dir) -> List[Dict]:
    os.makedirs(output_dir, exist_ok=True)
    frame_interval = int(round(fps / frame_per_second))
    frame_infos = []
    frame_id = 0
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:
            out_path = os.path.abspath(os.path.join(output_dir, f"{frame_id}.jpg"))
            cv2.imwrite(out_path, frame)
            frame_infos.append({"img_path": out_path, "frame": frame_id})
        frame_id += 1
    return frame_infos


def load_existing_frames(frames_dir: str) -> List[Dict]:
    frame_infos = []
    for fname in sorted(os.listdir(frames_dir), key=lambda x: int(os.path.splitext(x)[0]) if x.endswith(".jpg") else -1):
        if fname.endswith(".jpg"):
            path = os.path.abspath(os.path.join(frames_dir, fname))
            frame_id = int(os.path.splitext(fname)[0])
            frame_infos.append({"img_path": path, "frame": frame_id})
    return frame_infos


def copy_frames(img_paths: List[str], dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    for p in img_paths:
        shutil.copy2(p, os.path.join(dst_dir, os.path.basename(p)))


# ==================== 主流程 ====================

def process_single_video(
    video_path: str,
    output_root: str = "./tmp_meta",
    skip_extraction: bool = False,
):
    frames_dir = os.path.join(output_root, "frames")
    entropy_dir = os.path.join(output_root, "entropy_removed")
    phash_dir = os.path.join(output_root, "phash_removed")
    sscd_dir = os.path.join(output_root, "sscd_removed")
    meta_path = os.path.join(output_root, "meta.jsonl")
    stats_path = os.path.join(output_root, "stats.json")

    stats = StageStats()

    # ==================== 抽帧 ====================
    if skip_extraction and os.path.isdir(frames_dir) and os.listdir(frames_dir):
        print(f"[INFO] Skip extraction, loading from {frames_dir}")
        t0 = time.time()
        frame_infos = load_existing_frames(frames_dir)
        stats.record("Extraction", "Load Existing", time.time() - t0, len(frame_infos))
    else:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Video: {video_path} | FPS: {fps:.2f}")
        t0 = time.time()
        frame_infos = extract_frames(cap, fps, frame_per_second=1, output_dir=frames_dir)
        cap.release()
        stats.record("Extraction", "Frame Extraction", time.time() - t0, len(frame_infos))

    img_paths = [fi["img_path"] for fi in frame_infos]
    print(f"[INFO] Total frames: {len(img_paths)}")

    # ==================== Quality Filtering ====================
    pix_entropy = PixelEntropy(batch_size=256, num_workers=10, device="cuda")

    # Entropy Computation
    t0 = time.time()
    entropy_scores = pix_entropy.entropy_score_batch(img_paths)
    stats.record("Quality Filtering", "Entropy Computation", time.time() - t0, len(img_paths))

    # Edge Computation
    t0 = time.time()
    edge_scores = pix_entropy.edge_score_batch(img_paths)
    stats.record("Quality Filtering", "Edge Computation", time.time() - t0, len(img_paths))

    # Adaptive fusion + filtering
    entropy_arr = np.array(entropy_scores)
    edge_arr = np.array(edge_scores)
    dev = np.abs(np.stack([edge_arr, entropy_arr], axis=1) - 0.5)
    weights = dev / (dev.sum(axis=1, keepdims=True) + 1e-8)
    final_scores = weights[:, 0] * edge_arr + weights[:, 1] * entropy_arr

    img_paths_quality, meta_infos_quality, img_paths_quality_removed = [], [], []
    for fi, score, ent, edg in zip(frame_infos, final_scores, entropy_scores, edge_scores):
        if score >= 0.3:
            fi["entropy"] = round(float(ent), 3)
            fi["edge"] = round(float(edg), 3)
            fi["quality_score"] = round(float(score), 3)
            img_paths_quality.append(fi["img_path"])
            meta_infos_quality.append(fi)
        else:
            img_paths_quality_removed.append(fi["img_path"])

    copy_frames(img_paths_quality_removed, entropy_dir)
    stats.record("Quality Filtering", "Filtering", 0, len(img_paths_quality))
    print(f"[INFO] After quality: {len(img_paths_quality)} (removed {len(img_paths_quality_removed)})")

    # ==================== Perceptual Dedup (pHash) ====================
    phash_t = 1 - 14 / 64
    sscd_t = 0.75
    if len(img_paths_quality) >= 6000:
        phash_t += 2 / 64
        sscd_t += 0.05

    p_hash = PHash()

    # pHash Computation
    t0 = time.time()
    phash_vectors = p_hash.encode_images(img_paths_quality)
    stats.record("Perceptual Dedup", "pHash Computation", time.time() - t0, len(img_paths_quality))

    # pHash Index + CC
    unique_ids_phash = deduplicate_cc_with_stats(
        phash_vectors, "BinaryFlat", K=256, T=phash_t,
        stats=stats, stage_name="Perceptual Dedup", method_name="pHash"
    )

    img_paths_phash = [img_paths_quality[i] for i in unique_ids_phash]
    phash_set = set(unique_ids_phash)
    img_paths_phash_removed = [img_paths_quality[i] for i in range(len(img_paths_quality)) if i not in phash_set]
    copy_frames(img_paths_phash_removed, phash_dir)
    print(f"[INFO] After phash: {len(img_paths_phash)} (removed {len(img_paths_phash_removed)})")

    # ==================== Semantic Dedup (SSCD) ====================
    sscd_emb = Embedding(batch_size=256, num_workers=10, device="cuda")

    # SSCD Inference
    t0 = time.time()
    sscd_vectors = sscd_emb.encode_images(img_paths_phash)
    stats.record("Semantic Dedup", "SSCD Inference", time.time() - t0, len(img_paths_phash))

    # SSCD Index + CC
    unique_ids_sscd = deduplicate_cc_with_stats(
        sscd_vectors, "FlatIP", K=256, T=sscd_t,
        stats=stats, stage_name="Semantic Dedup", method_name="SSCD"
    )

    img_paths_sscd = [img_paths_phash[i] for i in unique_ids_sscd]
    meta_infos_sscd = [meta_infos_quality[unique_ids_phash[i]] for i in unique_ids_sscd]

    sscd_set = set(unique_ids_sscd)
    img_paths_sscd_removed = [img_paths_phash[i] for i in range(len(img_paths_phash)) if i not in sscd_set]
    copy_frames(img_paths_sscd_removed, sscd_dir)
    print(f"[INFO] After sscd: {len(img_paths_sscd)} (removed {len(img_paths_sscd_removed)})")

    # ==================== 保存结果 ====================
    with open(meta_path, "w", encoding="utf-8") as f:
        for meta in meta_infos_sscd:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    stats.save(stats_path)
    stats.print_summary()

    print(f"\nOutputs:")
    print(f"  Final frames:   {len(img_paths_sscd):>6}")
    print(f"  Meta:           {meta_path}")
    print(f"  Stats:          {stats_path}")


if __name__ == "__main__":
    video = "tmp_debug_frames/video/7202881362666258979.mp4"
    output_root = "./tmp_debug_frames"

    process_single_video(
        video_path=video,
        output_root=output_root,
        skip_extraction=True,
    )