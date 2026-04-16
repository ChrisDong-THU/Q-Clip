import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    """Base class for weighted losses."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, *args, **kwargs) -> torch.Tensor:
        loss = self.compute(*args, **kwargs)
        return self.weight * loss

    def compute(self, *args, **kwargs) -> torch.Tensor:
        """Override in subclass to compute raw loss."""
        raise NotImplementedError("Subclass must implement compute() method.")


class ContrastiveLoss(BaseLoss):
    """VRL-style binary contrastive loss."""

    def __init__(self, weight: float = 1.0, temperature: float = 0.07):
        super().__init__(weight)
        self.temperature = temperature

    def compute(self, anchor_feat: torch.Tensor, pos_feat: torch.Tensor) -> torch.Tensor:
        z_a = F.normalize(anchor_feat, dim=-1)  # [M, D]
        z_b = F.normalize(pos_feat, dim=-1)  # [M, D]

        logits = torch.mm(z_a, z_b.t()) / self.temperature  # [M, M]
        labels = torch.eye(z_a.shape[0], device=logits.device)  # diagonal = positive

        return F.binary_cross_entropy_with_logits(logits, labels)


class KoLeoLoss(BaseLoss):
    """KoLeo uniformity loss for slot features."""

    def __init__(self, weight: float = 1.0, eps: float = 1e-8):
        super().__init__(weight)
        self.eps = eps

    def compute(self, slot_feat: torch.Tensor, aid: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slot_feat: [B, 2, N, D] slot features from two clips
            aid: [B] album id
            pid: [B] pair id
        Returns:
            Scalar KoLeo loss (-log(nearest negative neighbor distance).)
        """
        B, C, D = slot_feat.shape

        feats = F.normalize(slot_feat.reshape(-1, D), dim=-1)
        M = feats.shape[0]
        if M <= 1:
            return feats.sum() * 0.0

        m_aid = aid.unsqueeze(1).expand(B, C).reshape(-1)
        m_pid = pid.unsqueeze(1).expand(B, C).reshape(-1)

        pos_mask = (
            (m_aid.unsqueeze(1) == m_aid.unsqueeze(0))
            & (m_pid.unsqueeze(1) == m_pid.unsqueeze(0))
        )

        neg_mask = ~pos_mask

        # 计算相似度并排除非负样本
        dots = torch.mm(feats, feats.t())
        dots = dots.masked_fill(~neg_mask, float("-inf"))

        # 寻找最近负样本
        max_vals, nn_indices = torch.max(dots, dim=1)
        valid = torch.isfinite(max_vals)
        if not valid.any():
            return feats.sum() * 0.0

        # 计算距离并求 loss
        anchor = feats[valid]
        neighbor = feats[nn_indices[valid]]
        distances = torch.norm(anchor - neighbor, dim=1)

        return -torch.log(distances + self.eps).mean()
