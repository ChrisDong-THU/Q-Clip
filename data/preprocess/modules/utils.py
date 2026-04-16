from typing import List, Optional, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    """Custom dataset for loading and transforming images"""
    
    def __init__(self, img_fps: List[str], transform: transforms.Compose):
        self.img_fps = img_fps
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.img_fps)
    
    def __getitem__(self, idx: int) -> Tuple[Optional[torch.Tensor], str]:
        """Load and transform image at index idx"""
        img_fp = self.img_fps[idx]
        try:
            image = Image.open(img_fp).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"[WARN] Failed to load image {img_fp}: {e}")
            return None