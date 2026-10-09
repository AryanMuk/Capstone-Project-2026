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
from src.Data_Preprocessing import load_celeba_array 

MODEL_NAME = "facenet-pytorch InceptionResnetV1 (vggface2)"


def get_device(prefer: Optional[str] = None) -> torch.device:
    """'cuda' or 'cpu' if requested, otherwise cuda when available."""
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_facenet(device: torch.device) -> torch.nn.Module:
    from facenet_pytorch import InceptionResnetV1 
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

def _signature(names: Sequence[str], cfg: PipelineConfig, shard_size: int) -> Dict:
    digest = hashlib.sha1("\n".join(names).encode()).hexdigest()
    return {"n": len(names), "names_sha1": digest, "shard_size": shard_size,
            "image_size": cfg.image_size, "celeba_crop": cfg.celeba_crop, "model": MODEL_NAME}


def _prepare_shard_dir(shard_dir: Path, signature: Dict) -> None:
    """Shards are only reusable if they were made from the same images and settings."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    sig_file = shard_dir / "signature.json"
    if sig_file.exists() and json.loads(sig_file.read_text()) != signature:
        raise RuntimeError(
            f"{shard_dir} holds shards from a different run (images or settings changed). "
            "Delete that folder to start over.")
    sig_file.write_text(json.dumps(signature, indent=2))


def generate(cfg: PipelineConfig, paths: Optional[Paths] = None, shard_size: int = 8192,
             device: Optional[torch.device] = None, fp16: bool = False,
             num_workers: Optional[int] = None, limit: Optional[int] = None,
             keep_shards: bool = False) -> Dict:
    paths = (paths or get_paths()).ensure()
    device = device or get_device()
    splits = load_splits(paths)
    names = splits["image"].tolist()[:limit] if limit else splits["image"].tolist()
    folder = images_dir(paths)

    shard_dir = paths.artifacts / ("emb_shards_limit" if limit else "emb_shards")
    _prepare_shard_dir(shard_dir, _signature(names, cfg, shard_size))
    model = load_facenet(device)

    n_shards = (len(names) + shard_size - 1) // shard_size
    done_images, resumed, t_start = 0, 0, time.time()
    for s in range(n_shards):
        shard_file = shard_dir / f"shard_{s:05d}.npy"
        if shard_file.exists():
            resumed += 1
            continue
        chunk = names[s * shard_size:(s + 1) * shard_size]
        parts = [embed_uint8(model, b, device, fp16)
                 for b in make_loader(folder, chunk, cfg, device, num_workers)]
        arr = np.concatenate(parts)
        tmp = shard_dir / f"shard_{s:05d}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, shard_file)                     
        done_images += len(chunk)
        rate = done_images / max(time.time() - t_start, 1e-9)
        print(f"  shard {s + 1}/{n_shards} done  ({rate:.0f} img/s)")

    emb = np.concatenate([np.load(shard_dir / f"shard_{s:05d}.npy") for s in range(n_shards)])
    if emb.shape != (len(names), cfg.embedding_dim):
        raise RuntimeError(f"unexpected embedding shape {emb.shape}")
    seconds = time.time() - t_start
    suffix = "_limit" if limit else ""
    np.save(paths.file(EMBEDDINGS_FILE.replace(".npy", f"{suffix}.npy")), emb)
    pd.DataFrame({"image": names, "row": np.arange(len(names))}).to_csv(
        paths.file(EMBEDDINGS_INDEX_FILE.replace(".csv", f"{suffix}.csv")), index=False)
    meta = {"model": MODEL_NAME, "n_images": len(names), "dim": int(emb.shape[1]),
            "device": str(device), "fp16": fp16, "seconds_this_run": round(seconds, 1),
            "images_embedded_this_run": done_images,
            "images_per_sec_this_run": round(done_images / seconds, 1) if done_images else None,
            "shards_total": n_shards, "shards_resumed": resumed,
            "torch": torch.__version__, "limit": limit}
    write_json(paths.file(f"embedding_meta{suffix}.json"), meta)
    if not keep_shards and not limit:
        for f in shard_dir.glob("shard_*.npy"):
            f.unlink()
    return meta

def sanity_check(emb: np.ndarray, splits: pd.DataFrame, n_pairs: int = 2000,
                 seed: int = 0) -> Dict:
    """Hard failures: NaN/inf or non-unit vectors. Soft warning: identities do not separate
    (usually means the crop/preprocessing is wrong)."""
    if not np.isfinite(emb).all():
        raise RuntimeError("embeddings contain NaN or inf")
    norms = np.linalg.norm(emb, axis=1)
    if np.abs(norms - 1.0).max() > 1e-3:
        raise RuntimeError(f"embeddings are not unit length (max deviation "
                           f"{np.abs(norms - 1).max():.4f})")
    rng = np.random.default_rng(seed)
    ident = splits["identity"].to_numpy()
    by_id = pd.Series(np.arange(len(ident))).groupby(ident).apply(np.array)
    multi = by_id[by_id.map(len) >= 2]
    gen = []
    for ids in multi.sample(min(n_pairs, len(multi)), random_state=seed, replace=True):
        a, b = rng.choice(ids, size=2, replace=False)
        gen.append(float(emb[a] @ emb[b]))
    i, j = rng.integers(0, len(ident), n_pairs), rng.integers(0, len(ident), n_pairs)
    keep = ident[i] != ident[j]
    imp = (emb[i[keep]] * emb[j[keep]]).sum(axis=1)
    out = {"genuine_mean": float(np.mean(gen)), "impostor_mean": float(imp.mean()),
           "separated": bool(np.mean(gen) > imp.mean() + 0.05)}
    if not out["separated"]:
        out["warning"] = ("genuine and impostor cosines barely differ; check the crop size and "
                          "that images are the aligned CelebA files")
    return out


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Generate FaceNet embeddings for every split image")
    p.add_argument("--shard-size", type=int, default=8192)
    p.add_argument("--fp16", action="store_true", help="half precision on GPU (faster)")
    p.add_argument("--workers", type=int, default=None, help="dataloader workers")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    p.add_argument("--limit", type=int, default=None, help="only the first N images (dry run)")
    p.add_argument("--keep-shards", action="store_true")
    args = p.parse_args(argv)

    cfg = load_pipeline_config()
    device = get_device(args.device)
    print(f"Device: {device}   model: {MODEL_NAME}")
    meta = generate(cfg, shard_size=args.shard_size, device=device, fp16=args.fp16,
                    num_workers=args.workers, limit=args.limit, keep_shards=args.keep_shards)
    print(f"Embedded {meta['images_embedded_this_run']} images in {meta['seconds_this_run']}s "
          f"({meta['images_per_sec_this_run']} img/s), resumed {meta['shards_resumed']} shards")

    paths = get_paths()
    suffix = "_limit" if args.limit else ""
    emb = np.load(paths.file(EMBEDDINGS_FILE.replace(".npy", f"{suffix}.npy")))
    splits = load_splits(paths).iloc[:len(emb)]
    check = sanity_check(emb, splits)
    print(f"Sanity: genuine cosine {check['genuine_mean']:.3f} vs impostor "
          f"{check['impostor_mean']:.3f}")
    if "warning" in check:
        print("WARNING:", check["warning"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
