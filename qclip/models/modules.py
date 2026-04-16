from typing import Literal

from torch import nn, Tensor
from torch.nn import MultiheadAttention as MHA


def _build_mlp(embed_dim: int, mid_dim: int, dropout: float = 0.0, activation: Literal["gelu", "relu"] = "gelu") -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(embed_dim, mid_dim),
        nn.GELU() if activation == "gelu" else nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(mid_dim, embed_dim),
        nn.Dropout(dropout),
    )


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, mlp_ratio: float = 4.0, activation: Literal["gelu", "relu"] = "gelu"):
        super().__init__()
        self.attn = MHA(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mid_dim = int(embed_dim * mlp_ratio)
        self.mlp = _build_mlp(embed_dim, mid_dim, dropout, activation)

        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, q: Tensor, k: Tensor, v: Tensor, need_weights: bool = False) -> Tensor:
        attn_out, attn_weights = self.attn(q, k, v, need_weights=need_weights, average_attn_weights=True)
        
        q = q + self.dropout(attn_out)
        q = self.norm1(q)
        q = self.norm2(q + self.mlp(q))

        return q, attn_weights


class FeedForwardBlock(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        dropout: float = 0.0, 
        mlp_ratio: float = 4.0, 
        activation: Literal["gelu", "relu"] = "gelu",
        num_blocks: int = 2,
        pre_norm: bool = True
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.norm = nn.LayerNorm(embed_dim)
        
        mid_dim = int(embed_dim * mlp_ratio)

        self.up_proj = nn.Sequential(
            nn.Linear(embed_dim, mid_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(mid_dim),
                nn.Linear(mid_dim, mid_dim),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_blocks - 1)
        ])

        self.down_proj = nn.Sequential(
            nn.Linear(mid_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        if self.pre_norm:
            x = self.norm(x)

        x = self.up_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.down_proj(x)

        if self.pre_norm:
            return identity + x
        else:
            return self.norm(identity + x)
