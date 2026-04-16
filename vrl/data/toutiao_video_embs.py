from typing import Tuple, List, Dict, Any
import json
import random
import fsspec
import webdataset as wds
from torch.utils.data import IterableDataset

from .partitions import FramesPartition


class ToutiaoVideoEmbs(IterableDataset):
    def __init__(
        self,
        root: str | List[str],
        repeat: bool = False,
        resample: bool = False,
        shardshuffle: bool | int = False,
        partition_cfg: Dict[str, int] = {"part_size": 8, "embed_dim": 1536},
        buffer_size: int = 1000,
    ) -> None:
        super().__init__()
        
        self.root = root
        
        assert not (repeat and resample), "repeat and resample cannot be True at the same time."
        self.repeat = repeat
        self.resample = resample
        self.shardshuffle = shardshuffle

        self.partition = FramesPartition(**partition_cfg)

        self.buffer_size = buffer_size

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
        metadata = json.loads(sample["json"].decode('utf-8'))
        aid, pid = int(metadata["album_id"]), int(metadata["pair_id"])

        pairs, pairs_mask = self.partition(sample["npy"])
        
        return {"pairs": pairs, "pairs_mask": pairs_mask, "aid": aid, "pid": pid}

    def __iter__(self):
        buffer = []

        for sample in self._dataset:  # infinite stream
            buffer.append(sample)

            if len(buffer) >= self.buffer_size:
                random.shuffle(buffer)

                try:
                    output = self._process_sample(buffer.pop())
                    yield output
                except Exception as e:
                    print(f"Skip corrupted sample: {e}")
                    continue

