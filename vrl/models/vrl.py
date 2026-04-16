import lightning as L
from timm.optim import create_optimizer_v2 as create_optimizer
from timm.scheduler import create_scheduler_v2 as create_scheduler
import torch
from torch import nn, Tensor

from .modules import AttentionBlock
from ..loss import ContrastiveLoss, KoLeoLoss


class VRL(L.LightningModule):
    def __init__(self, embed_dim: int, enc_depth: int = 1, attn_cfg: dict = None, 
                 optim_cfg: dict = None, lr_scheduler_cfg: dict = None, loss_cfg: dict = None, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim

        self.enc_depth = enc_depth
        self.enc_blocks = nn.ModuleList([
            AttentionBlock(**attn_cfg) for _ in range(enc_depth)
        ])
        
        self.optim_cfg = optim_cfg
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.automatic_optimization = False

        self.contrastive_loss = ContrastiveLoss(**loss_cfg["contrastive"])
        # self.koleo_loss = KoLeoLoss(**loss_cfg["koleo"])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.enc_blocks:
            x, _ = block(q=x, k=x, v=x)
        
        # average pooling
        x = torch.mean(x, dim=1)
        
        return x
    
    def configure_optimizers(self):
        optimizer = create_optimizer(self.parameters(), **self.optim_cfg)
        scheduler = create_scheduler(optimizer=optimizer, **self.lr_scheduler_cfg)[0]

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def training_step(self, batch, batch_idx):
        pairs, pairs_mask = batch["pairs"], batch["pairs_mask"]
        aid, pid = batch["aid"], batch["pid"]
        
        all_clips = pairs.flatten(0, 1) # [B*2, T, D_in]
        all_clips_mask = pairs_mask.flatten(0, 1) # [B*2, T', D_in]

        feats = self(all_clips)  # [B*2, D]
        feats_mask = self(all_clips_mask)  # [B*2, D]

        feats = feats.reshape(-1, 2, feats.shape[-1])
        feats_mask = feats_mask.reshape(-1, 2, feats_mask.shape[-1])

        # Losses
        loss1 = self.contrastive_loss(feats[:, 0], feats[:, 1])
        loss2 = self.contrastive_loss(feats[:, 0], feats_mask[:, 1])
        loss3 = self.contrastive_loss(feats_mask[:, 0], feats[:, 1])

        loss_contrastive = loss1 + loss2 + loss3
        loss_koleo = self.koleo_loss(feats, aid, pid)

        loss_total = loss_contrastive + loss_koleo

        # Optimization
        optimizer, scheduler = self.optimizers(), self.lr_schedulers()
        optimizer.zero_grad()
        self.manual_backward(loss_total)
        optimizer.step()
        scheduler.step_update(self.global_step)

        # Logging
        self.log("train/loss_contrastive", loss_contrastive, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train/loss_koleo", loss_koleo, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train/loss_total", loss_total, on_step=True, on_epoch=False, prog_bar=True)

