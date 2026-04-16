from typing import Callable, Optional, Tuple, List, Any
import os
from enum import Enum
import json

import fsspec
import webdataset as wds
from torch.utils.data import IterableDataset

from .decoders import ImageDataDecoder, TargetDecoder
from ..transforms import BorderCropper


class ToutiaoVideos(IterableDataset):
    def __init__(
        self,
        *,
        root: str | List[str],
        repeat: bool = False,
        resample: bool = False,
        shardshuffle: bool | int = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        border_crop: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.root = root

        assert not (repeat and resample), "repeat and resample cannot be True at the same time."
        self.repeat = repeat
        self.resample = resample
        self.shardshuffle = shardshuffle

        self.transform = transform
        self.target_transform = target_transform
        self.border_cropper = BorderCropper() if border_crop else lambda x: x

        self.image_decoder = ImageDataDecoder
        self.target_decoder = TargetDecoder

        self._dataset = self._build_pipeline()

    def _get_tar_files(self) -> List[str]:
        fs = fsspec.filesystem('hdfs')

        if isinstance(self.root, str):
            tar_files = fs.glob(f"{self.root}/*.tar")
        else:
            tar_files = []
            for r in self.root:
                tar_files.extend(fs.glob(f"{r}/*.tar"))
        
        return tar_files
    
    def _build_pipeline(self) -> wds.WebDataset:
        tar_files = self._get_tar_files()
        
        if not tar_files:
            raise RuntimeError(f'No tar files found in: "{self.root}"')
        else:
            print(f"Use {len(tar_files)} wds files.")
        
        tar_urls = [f"pipe:hdfs dfs -cat /{f}" for f in tar_files]
        
        dataset = wds.WebDataset(
            tar_urls,
            resampled=self.resample,
            shardshuffle=self.shardshuffle,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            handler=wds.handlers.warn_and_continue,  # 忽略损坏样本
        )

        if self.resample or self.shardshuffle:
            dataset = dataset.shuffle(4000)

        if self.repeat:
            dataset = dataset.repeat()

        return dataset

    def _process_sample(self, sample: dict) -> Tuple[Any, Any]:
        """处理单个样本."""
        image_data = sample["jpg"]
        image = self.image_decoder(image_data).decode()

        target = json.loads(sample["json"].decode('utf-8'))
        target = self.target_decoder(target).decode()
        
        image = self.border_cropper(image)
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target

    def __iter__(self):
        for sample in self._dataset:
            try:
                yield self._process_sample(sample)
            except Exception as e:
                print(f"Skip corrupted sample: {e}")
                continue

