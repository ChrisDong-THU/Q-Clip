import numpy as np
import torch
import random


class FramesPartition:    
    def __init__(self, num_parts: int = 2, part_size: int = 256, interval_size: int = 512, embed_dim: int = 1536):
        self.num_parts = num_parts
        self.part_size = part_size
        self.interval_size = interval_size
        self.embed_dim = embed_dim

        print(f"[INFO] Using partition with num_parts={self.num_parts}, part_size={self.part_size}, interval_size={self.interval_size}")

    def __call__(self, emb_data: bytes) -> torch.Tensor:
        emb = np.frombuffer(emb_data, dtype=np.float32).copy()
        emb = torch.from_numpy(emb).reshape(-1, self.embed_dim)

        start = random.randint(0, emb.shape[0] - self.interval_size)
        emb = emb[start:start+self.interval_size]
        
        rand_matrix = torch.rand(self.num_parts, self.interval_size)
        indices = rand_matrix.argsort(dim=-1)[:, :self.part_size]

        pairs = emb[indices]
        
        return pairs