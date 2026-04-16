import numpy as np
import torch
import random


class FramesPartition:
    def __init__(self, part_size: int = 8, mask_size: int = 2, r_plus_range: tuple = (1.0, 2.0), embed_dim: int = 1536):
        self.part_size = part_size
        self.mask_size = mask_size
        self.r_plus_range = r_plus_range
        self.embed_dim = embed_dim
        self.half = part_size // 2
        # pos clip 最大扩展范围（用于计算 anchor 的安全边界）
        self.margin = int(self.half * r_plus_range[1]) + 1

    def __call__(self, emb_data: bytes):
        emb = torch.from_numpy(np.frombuffer(emb_data, dtype=np.float32).copy())
        emb = emb.reshape(-1, self.embed_dim)  # [T, D]
        T, N = len(emb), self.part_size

        # 1. Anchor: 随机取连续 N 帧（预留 margin 防止 pos clip 越界）
        start = random.randint(self.margin, T - N - self.margin)
        anchor_idx = torch.arange(start, start + N)

        # 2. Positive: 以 anchor 中随机一帧为中心，间隔 r+ 采样
        median = anchor_idx[random.randint(0, N - 1)].item()
        r_plus = random.uniform(*self.r_plus_range)
        offsets = torch.arange(-self.half, N - self.half, dtype=torch.float32)
        pos_idx = (median + offsets * r_plus).round().long().clamp(0, T - 1)

        # 取 embedding
        anchor_emb = emb[anchor_idx]  # [N, D]
        pos_emb = emb[pos_idx]        # [N, D]

        # Mask
        keep = N - self.mask_size
        anchor_mask = anchor_emb[torch.randperm(N)[:keep].sort().values]
        pos_mask = pos_emb[torch.randperm(N)[:keep].sort().values]

        return torch.stack([anchor_emb, pos_emb]), torch.stack([anchor_mask, pos_mask])


if __name__ == "__main__":
    # ========== 参数配置 ==========
    embed_dim = 1536
    num_frames = 100  # 模拟视频帧数
    part_size = 8
    mask_size = 2
    
    # ========== 构造模拟数据 ==========
    # 模拟 num_frames 帧的 embedding，每帧 embed_dim 维
    fake_emb = np.random.randn(num_frames, embed_dim).astype(np.float32)
    emb_data = fake_emb.tobytes()  # 转为 bytes，模拟实际输入
    
    print(f"[TEST] 输入 emb_data: {len(emb_data)} bytes")
    print(f"[TEST] 对应帧数: {num_frames}, embed_dim: {embed_dim}")
    
    # ========== 初始化并调用 ==========
    partition = FramesPartition(
        part_size=part_size,
        mask_size=mask_size,
        r_plus_range=(0.5, 2.0),
        embed_dim=embed_dim
    )
    
    pairs, pairs_mask = partition(emb_data)
    
    # ========== 验证输出形状 ==========
    print(f"\n[RESULT] pairs.shape: {pairs.shape}")           # 期望 [2, part_size, embed_dim]
    print(f"[RESULT] pairs_mask.shape: {pairs_mask.shape}")  # 期望 [2, part_size - mask_size, embed_dim]
    
    assert pairs.shape == (2, part_size, embed_dim), f"pairs 形状错误: {pairs.shape}"
    assert pairs_mask.shape == (2, part_size - mask_size, embed_dim), f"pairs_mask 形状错误: {pairs_mask.shape}"
    
    print("\n✓ 所有测试通过!")
    
    # ========== 多次调用测试随机性 ==========
    print("\n[TEST] 多次调用验证随机性:")
    for i in range(3):
        p, pm = partition(emb_data)
        # 打印 anchor 片段的第一帧前 5 维，验证每次不同
        print(f"  Run {i+1}: anchor[0][:5] = {p[0, 0, :5].tolist()}")