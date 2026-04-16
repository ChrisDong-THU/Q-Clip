from typing import Tuple
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn

from vrl.models import VRL


def get_default_config() -> DictConfig:
    p = Path(__file__).parent / "configs" / "default_config.yaml"
    return OmegaConf.load(p)


def get_config(config_path: str, strict: bool = True, output_dir: str = None) -> DictConfig:
    config = OmegaConf.load(config_path)
    default_config = get_default_config()
    
    config = OmegaConf.merge(default_config, config)
    if strict:
        OmegaConf.set_struct(config, True)
    
    if output_dir is not None:
        OmegaConf.save(config, Path(output_dir) / "config.yaml")
    
    return config


def build_vrl_model_for_eval(cfg_path: str = "config.yaml", ckpt_path: str = "step=20000.ckpt") -> Tuple[nn.Module, torch.dtype]:
    config = OmegaConf.load(cfg_path)
    model = VRL(**config.model)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    model.cuda()

    print(f"[INFO] Loaded VRL model from {ckpt_path}")
    return model, torch.float32
