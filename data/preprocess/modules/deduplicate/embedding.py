from typing import List
from pathlib import Path
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ..utils import ImageDataset


class Embedding:
    def __init__(self, batch_size: int = 32, num_workers: int = 10, device: str = "cuda"):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = self._get_transform()
        self.device = device
        self.model = self._load_model(device)
        
    def _load_model(self, device: str = "cuda") -> torch.nn.Module:
        """Load the embedding model from checkpoint."""
        model = torch.jit.load(Path(__file__).parent / "pretrained/sscd_disc_mixup.torchscript.pt")
        model.eval()
        return model.to(device)
    
    def _get_transform(self) -> transforms.Compose:
        """Get image transformation pipeline for SSCD model."""
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _load_image(self, img_fp: str) -> torch.Tensor:
        """Helper function to load and transform a single image (for parallel processing)"""
        try:
            image = Image.open(img_fp).convert("RGB")
            return self.transform(image)
        except Exception as e:
            print(f"[WARN] Failed to load image {img_fp}: {e}")
            return None
    
    @torch.no_grad()
    def encode_image(self, img_fp: str) -> np.ndarray:
        """Encode single image file to embedding vector"""
        # Load and preprocess image
        image = self._load_image(img_fp)
        if image is None:
            return np.array([])
        
        image_tensor = image.unsqueeze(0).to(self.device)
        embedding = self.model(image_tensor)

        return embedding.squeeze(0).cpu().numpy()
    
    @torch.no_grad()
    def encode_images(self, img_fps: List[str]) -> np.ndarray:
        """Encode multiple image files to embedding vectors using DataLoader for efficient batch processing"""
        all_embeddings = []
        
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
            embeddings = self.model(x)
            all_embeddings.append(embeddings.cpu().numpy())

        if not all_embeddings:
            return np.array([])
            
        return np.concatenate(all_embeddings, axis=0)
        