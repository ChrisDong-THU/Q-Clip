import lightning as L
from timm.optim import create_optimizer_v2 as create_optimizer
from timm.scheduler import create_scheduler_v2 as create_scheduler
import random
import torch
from torch import nn, Tensor

from .modules import FeedForwardBlock, AttentionBlock
from ..loss import InfoNCELoss, KoLeoLoss


class QClip(L.LightningModule):
    def __init__(self, feat_dim: int, embed_dim: int, local_clip_len: tuple = (32, 64, 128), enc_depth: int = 0, dec_depth: int = 4, query_num: int = 1, query_grad: bool = True, 
                 attn_cfg: dict = None, ffn_cfg: dict = None, optim_cfg: dict = None, lr_scheduler_cfg: dict = None, loss_cfg: dict = None, **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.query_num = query_num

        self.proj_in = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        self.enc_depth = enc_depth
        self.enc_blocks = nn.ModuleList([
            AttentionBlock(**attn_cfg) for _ in range(enc_depth)
        ])
        
        self.dec_depth = dec_depth
        self.dec_blocks = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": AttentionBlock(**attn_cfg),
                "ffn": FeedForwardBlock(**ffn_cfg),
                "cross_attn": AttentionBlock(**attn_cfg),
            })
            for _ in range(dec_depth)
        ])
        
        self.query = nn.Parameter(torch.randn(1, query_num, embed_dim), requires_grad=query_grad)
        self.query_pos = nn.Parameter(torch.randn(1, query_num, embed_dim), requires_grad=True)

        self.local_clip_len = local_clip_len

        self.optim_cfg = optim_cfg
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.automatic_optimization = False

        self.infonce_loss = InfoNCELoss(**loss_cfg["infonce"])
        self.koleo_loss = KoLeoLoss(**loss_cfg["koleo"])

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.query_pos)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Frame CLS embeddings of shape [B, M, D], where:
                - B: batch size
                - M: number of frames per partition
                - D: embedding dimension

        Returns:
            query (Tensor): Shape [B, N, D]
            attn_weights (Tensor): Shape [B, N, M]
        """
        B, N, D = x.shape
        
        x = self.proj_in(x)
        for block in self.enc_blocks:  # default enc_depth = 0
            x, _ = block(q=x, k=x, v=x)
        
        query = self.query.expand(B, -1, -1)
        query_pos = self.query_pos.expand(B, -1, -1)
        
        for i, block in enumerate(self.dec_blocks):
            need_weights = (i == self.dec_depth - 1) # only return attn_weights for the last block
            query, _ = block["self_attn"](q=query+query_pos, k=query+query_pos, v=query)
            query, attn_weights = block["cross_attn"](q=query+query_pos, k=x, v=x, need_weights=need_weights)
            query = block["ffn"](query)
        
        return query, attn_weights
    
    def configure_optimizers(self):
        optimizer = create_optimizer(self.parameters(), **self.optim_cfg)
        scheduler = create_scheduler(optimizer=optimizer, **self.lr_scheduler_cfg)[0]

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def training_step(self, batch, batch_idx):
        pairs = batch["pairs"]  # [B, 2, T, D_in]
        aid, pid = batch["aid"], batch["pid"]
        
        global_clips, local_clips = pairs[:, 0], pairs[:, 1]

        local_clip_len = random.choice(self.local_clip_len)
        local_clips = local_clips[:, :local_clip_len]

        global_feats, _ = self(global_clips)  # [B, N, D]
        local_feats, _ = self(local_clips)  # [B, N, D]
        feats = torch.stack([global_feats, local_feats], dim=1)  # [B, 2, N, D]

        # Losses
        loss_koleo = self.koleo_loss(feats, aid, pid)
        loss_infonce = self.infonce_loss(feats, aid, pid)

        loss_total = loss_koleo + loss_infonce

        # Optimization
        optimizer, scheduler = self.optimizers(), self.lr_schedulers()
        optimizer.zero_grad()
        self.manual_backward(loss_total)
        optimizer.step()
        scheduler.step_update(self.global_step)

        # Logging
        self.log_dict({
            "train/loss_koleo": loss_koleo,
            "train/loss_infonce": loss_infonce,
        }, logger=True, on_step=True, on_epoch=False, prog_bar=False)

        self.log("train/loss_total", loss_total, on_step=True, on_epoch=False, prog_bar=True)

