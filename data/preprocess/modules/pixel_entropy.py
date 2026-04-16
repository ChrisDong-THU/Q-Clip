from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from .utils import ImageDataset


class PixelEntropy:
    def __init__(self, is_adaptive=True, bins=512, batch_size=32, num_workers=10, device="cuda"):
        super().__init__()
        self.is_adaptive = is_adaptive
        self.bins = bins
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = self._get_transform()
        
        self.device = device

        # Sobel kernels
        self.sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device)[None, None]
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device)[None, None]

        # Sobel 最大理论响应 = 4*sqrt(2) ≈ 5.66（输入 [0,1] 时）
        self.max_grad = 4.0 * 1.4142
        
    def _get_transform(self) -> transforms.Compose:
        """Get image transformation pipeline for pixel entropy."""
        return transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
        ])

    def edge_score(self, x: torch.Tensor):
        """x: Bx1xHxW"""
        # 使用 replicate padding 代替 zero padding，避免边界伪梯度
        x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
        gx = F.conv2d(x_pad, self.sobel_x, padding=0)
        gy = F.conv2d(x_pad, self.sobel_y, padding=0)

        grad = torch.sqrt(gx * gx + gy * gy) / self.max_grad

        # edge_density = ratio of non-zero gradient pixels
        edge_mask = grad > 1e-6
        edge_density = edge_mask.float().mean(dim=[1, 2, 3])

        # mean edge strength
        nonzero = torch.where(edge_mask, grad, torch.zeros_like(grad))
        edge_strength = nonzero.sum(dim=[1,2,3]) / (edge_mask.sum(dim=[1,2,3]).float() + 1e-6)

        return (edge_density + edge_strength) / 2

    def entropy_score(self, x: torch.Tensor):
        """计算 Shannon entropy (0~1)"""
        B = x.shape[0]

        # discretize to batch hist
        bin_edges = torch.linspace(0, 1, self.bins + 1, device=x.device)
        digitized = torch.bucketize(x.flatten(1), bin_edges)  # BxN, values 1~bins+1
        digitized = digitized.clamp(1, self.bins) - 1 # 0~bins-1

        # Prepare histogram
        hist = torch.zeros((B, self.bins), device=x.device, dtype=digitized.dtype)
        hist.scatter_add_(1, digitized, torch.ones_like(digitized))

        # prob: B x bins
        prob = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)

        # mask zeros to avoid nan
        mask = prob > 0
        logp = torch.zeros_like(prob)
        logp[mask] = torch.log2(prob[mask])
        entropy = -(prob * logp).sum(dim=1)  # shape: (B,)

        return entropy / 8.0 # normalize to 0-1 (log2(512)=9)

    @torch.no_grad()
    def encode_images(self, img_fps: List[str]) -> np.ndarray:
        edge_scores = []
        entropy_scores = []
        
        dataset = ImageDataset(img_fps, transform=self.transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False,
            drop_last=False
        )

        for x in dataloader:
            x = x.to(self.device)
            edge = self.edge_score(x)
            entropy = self.entropy_score(x)
            edge_scores.append(edge.cpu().numpy())
            entropy_scores.append(entropy.cpu().numpy())

        edge = np.concatenate(edge_scores, axis=0)
        entropy = np.concatenate(entropy_scores, axis=0)

        if self.is_adaptive:
            dev = np.abs(np.stack([edge, entropy], axis=1) - 0.5)    # [B,2]
            weights = dev / (dev.sum(axis=1, keepdims=True) + 1e-8)  # [B,2]
            score = (weights[:, 0] * edge + weights[:, 1] * entropy)
        else:
            score = 0.5 * (edge + entropy)
        
        return score

    @torch.no_grad()
    def entropy_score_batch(self, img_fps: List[str]) -> List[float]:
        """单独计算 entropy scores"""
        dataset = ImageDataset(img_fps, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        scores = []
        for x in dataloader:
            x = x.to(self.device)
            scores.append(self.entropy_score(x).cpu().numpy())
        return np.concatenate(scores, axis=0).tolist()

    @torch.no_grad()
    def edge_score_batch(self, img_fps: List[str]) -> List[float]:
        """单独计算 edge scores"""
        dataset = ImageDataset(img_fps, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        scores = []
        for x in dataloader:
            x = x.to(self.device)
            scores.append(self.edge_score(x).cpu().numpy())
        return np.concatenate(scores, axis=0).tolist()
