from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common.artifacts import (EMBEDDINGS_FILE, EMBEDDINGS_INDEX_FILE, images_dir, load_splits,
                              write_json)
from common.config import PipelineConfig, Paths, get_paths, load_pipeline_config
from src.Data_Preprocessing_Aryan import load_celeba_array 

MODEL_NAME = "facenet-pytorch InceptionResnetV1 (vggface2)"


def get_device(prefer: Optional[str] = None) -> torch.device:
    """'cuda' or 'cpu' if requested, otherwise cuda when available."""
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_facenet(device: torch.device) -> torch.nn.Module:
    from facenet_pytorch import InceptionResnetV1   # lazy: only needed when embedding
    return InceptionResnetV1(pretrained="vggface2").eval().to(device)


@torch.no_grad()
def embed_uint8(model: torch.nn.Module, batch: torch.Tensor, device: torch.device,
                fp16: bool = False) -> np.ndarray:
    """batch: uint8 tensor (N, H, W, 3) -> float32 array (N, 512), unit length.
    Standardisation (x - 127.5) / 128 is what facenet-pytorch's `fixed_image_standardization`
    applies, done on the device to keep the CPU free for JPEG decoding."""
    x = batch.to(device, non_blocking=True).permute(0, 3, 1, 2).float()
    x = (x - 127.5) / 128.0
    use_amp = fp16 and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        emb = model(x)
    emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
    return emb.cpu().numpy().astype(np.float32) 

class CelebADataset(Dataset):
    """Yields uint8 tensors (H, W, 3) produced by the same crop used everywhere else."""

    def __init__(self, folder: Path, names: Sequence[str], cfg: PipelineConfig):
        self.folder, self.names, self.cfg = Path(folder), list(names), cfg

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(load_celeba_array(self.folder / self.names[i], self.cfg))


def make_loader(folder: Path, names: Sequence[str], cfg: PipelineConfig, device: torch.device,
                num_workers: Optional[int]) -> DataLoader:
    workers = min(os.cpu_count() or 1, 8) if num_workers is None else num_workers
    return DataLoader(CelebADataset(folder, names, cfg), batch_size=cfg.batch_size,
                      shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")