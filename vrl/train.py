import argparse
import os

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from vrl.models.vrl import VRL
from vrl.data.from_config import DataModuleFromConfig
from vrl.utils import get_config


def get_args():
    parser = argparse.ArgumentParser(description="VRL Training")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="test-1",
        help="Config file name without extension (located in vrl/configs/ directory).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/vrl_vitg16_toutiao",
        help="Directory to save logs.",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="debug",
        help="Experiment name.",
    )
    return parser.parse_args()


def main(args):
    exp_dir = os.path.join(args.log_dir, args.exp)
    os.makedirs(exp_dir, exist_ok=True)
    
    cfg_path = os.path.join("vrl/configs", f"{args.config}.yaml")
    cfg = get_config(cfg_path, strict=True, output_dir=exp_dir)
    
    # resume from checkpoint if specified
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")
        print(f"Resuming training from checkpoint: {ckpt_path}")

    dm = DataModuleFromConfig(**cfg.data)

    model = VRL(**cfg.model)

    trainer = L.Trainer(
        **cfg.trainer,
        callbacks=[
            ModelCheckpoint(
                dirpath=exp_dir,
                filename="{step}",
                **cfg.checkpoint,
            ),
            LearningRateMonitor(logging_interval='step'),
        ],
        logger=WandbLogger(
            project="vrl",
            name=args.exp
        ),
    )

    trainer.fit(model, dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    args = get_args()
    L.seed_everything(args.seed)
    main(args)