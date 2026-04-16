from typing import List
import numpy as np
from phash_cuda import phash


class PHash:
    def __init__(self, resize: int = 32, low_freq: int = 8):
        self.resize = resize
        self.low_freq = low_freq
        
    def encode_images(self, img_fps: List[str]) -> np.ndarray:
        hashes = phash(img_fps, self.resize, self.low_freq)
        hashes = np.array(hashes, dtype=np.uint8)
        
        return hashes