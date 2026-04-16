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


class InfoNCELoss(BaseLoss):
    """Symmetric InfoNCE loss for matched slot features across two clips."""

    def __init__(self, weight: float = 1.0, temperature: float = 0.07, slot_wise: bool = False):
        super().__init__(weight)
        self.temperature = temperature
        self.slot_wise = slot_wise

    def compute(
        self,
        slot_feat: torch.Tensor,
        aid: torch.Tensor,
        pid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            slot_feat: [B, 2, N, D] slot features from two clips
            aid: [B], album id
            pid: [B], pair id
        Returns:
            Scalar InfoNCE loss (symmetric)
        """
        B, _, N, D = slot_feat.shape
        
        queries_a = F.normalize(slot_feat[:, 0].reshape(-1, D), dim=-1)  # [B*N, D]
        queries_b = F.normalize(slot_feat[:, 1].reshape(-1, D), dim=-1)  # [B*N, D]

        M = queries_a.shape[0]
        if M == 0:
            return slot_feat.sum() * 0.0  # 无匹配 slot，返回 0 且保留计算图

        m_aid = aid.unsqueeze(1).expand(B, N).reshape(-1)                # [M]
        m_pid = pid.unsqueeze(1).expand(B, N).reshape(-1)                # [M]
        m_sid = torch.arange(N, device=slot_feat.device).repeat(B)       # [M]

        pos_mask = (
            (m_aid.unsqueeze(1) == m_aid.unsqueeze(0))
            & (m_pid.unsqueeze(1) == m_pid.unsqueeze(0))
            & (m_sid.unsqueeze(1) == m_sid.unsqueeze(0))
        )  # [M, M]

        if self.slot_wise:
            same_slot = (m_sid.unsqueeze(1) == m_sid.unsqueeze(0))
        else:
            same_slot = torch.ones_like(pos_mask, dtype=torch.bool)

        # 计算相似度矩阵
        logits_ab = torch.mm(queries_a, queries_b.t()) / self.temperature
        logits_ba = torch.mm(queries_b, queries_a.t()) / self.temperature

        loss_ab = self._multi_pos_nce(logits_ab, pos_mask, same_slot)
        loss_ba = self._multi_pos_nce(logits_ba, pos_mask, same_slot)

        return (loss_ab.mean() + loss_ba.mean()) / 2.0
    
    @staticmethod
    def _multi_pos_nce(logits: torch.Tensor, pos_mask: torch.Tensor, same_slot: torch.Tensor) -> torch.Tensor:
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()  # 数值稳定

        exp_logits = torch.exp(logits)
        exp_logits = exp_logits * same_slot  # 排除不同 slot

        denom = exp_logits.sum(dim=1, keepdim=True)
        log_prob = logits - torch.log(denom + 1e-8)

        pos_mask_float = pos_mask.float()
        num_pos = pos_mask_float.sum(dim=1).clamp(min=1.0)
        mean_log_prob_pos = (log_prob * pos_mask_float).sum(dim=1) / num_pos

        return -mean_log_prob_pos


class KoLeoLoss(BaseLoss):
    """koleo uniformity loss for slot features."""

    def __init__(self, weight: float = 1.0, eps: float = 1e-8, slot_wise: bool = False):
        super().__init__(weight)
        self.eps = eps
        self.slot_wise = slot_wise

    def compute(self, slot_feat: torch.Tensor, aid: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slot_feat: [B, 2, N, D] slot features from two clips
            aid: [B] album id
            pid: [B] pair id
        Returns:
            Scalar koleo loss (-log(nearest negative neighbor distance).)
        """
        B, C, N, D = slot_feat.shape
        device = slot_feat.device

        feats = F.normalize(slot_feat.reshape(-1, D), dim=-1)
        M = feats.shape[0]
        if M <= 1:
            return feats.sum() * 0.0

        m_aid = aid.unsqueeze(1).expand(B, C * N).reshape(-1)
        m_pid = pid.unsqueeze(1).expand(B, C * N).reshape(-1)
        m_sid = torch.arange(N, device=device).repeat(B * C)

        pos_mask = (
            (m_aid.unsqueeze(1) == m_aid.unsqueeze(0))
            & (m_pid.unsqueeze(1) == m_pid.unsqueeze(0))
            & (m_sid.unsqueeze(1) == m_sid.unsqueeze(0))
        )

        if self.slot_wise:
            same_slot = (m_sid.unsqueeze(1) == m_sid.unsqueeze(0))
            neg_mask = (~pos_mask) & same_slot
        else:
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

