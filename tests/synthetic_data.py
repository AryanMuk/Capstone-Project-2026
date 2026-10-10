from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image


def make_fake_celeba(root: Path, n_identities: int = 30, images_range: Tuple[int, int] = (10, 14),
                     seed: int = 0, make_zip: bool = False) -> Path:
    """Create root/img_align_celeba/NNNNNN.jpg and root/identity_CelebA.txt. Returns root
    (or the zip path when make_zip=True)."""
    rng = np.random.default_rng(seed)
    img_dir = root / "img_align_celeba"
    img_dir.mkdir(parents=True, exist_ok=True)
    lines, n = [], 0
    for ident in range(1, n_identities + 1):
        base = rng.integers(0, 255, size=(7, 6, 3), dtype=np.uint8)
        base_img = np.asarray(Image.fromarray(base).resize((178, 218), Image.BILINEAR),
                              dtype=np.float32)
        for _ in range(int(rng.integers(images_range[0], images_range[1] + 1))):
            n += 1
            noisy = np.clip(base_img + rng.normal(0, 12, base_img.shape), 0, 255).astype(np.uint8)
            name = f"{n:06d}.jpg"
            Image.fromarray(noisy).save(img_dir / name, quality=90)
            lines.append(f"{name} {1000 + ident}")
    (root / "identity_CelebA.txt").write_text("\n".join(lines) + "\n")
    if not make_zip:
        return root
    zip_path = root.parent / f"{root.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for f in sorted(root.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(root))
    return zip_path


def make_synthetic_embeddings(splits: pd.DataFrame, dim: int = 512, noise: float = 0.9,
                              seed: int = 0) -> np.ndarray:
    """One unit vector per row of `splits`: identity centre + Gaussian noise, re-normalised.
    Larger `noise` -> harder problem. Row i corresponds to splits.iloc[i]."""
    rng = np.random.default_rng(seed)
    idents = np.sort(splits["identity"].unique())
    centres = rng.standard_normal((len(idents), dim)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    pos = np.searchsorted(idents, splits["identity"].to_numpy())
    emb = centres[pos] + noise / np.sqrt(dim) * rng.standard_normal((len(splits), dim)).astype(
        np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb.astype(np.float32)


def write_synthetic_artifacts(paths, n_identities: int = 300, images_range: Tuple[int, int] = (12, 20),
                              noise: float = 0.9, seed: int = 0, cfg=None):
    """Write manifest/splits/config/embeddings exactly as the real stages would, but from
    synthetic embeddings, so Encryption -> Evaluation -> Security -> Backend can be tested
    without images or a model. Returns (splits, embeddings, cfg)."""
    from common.artifacts import (EMBEDDINGS_FILE, EMBEDDINGS_INDEX_FILE, MANIFEST_FILE,
                                  SPLITS_FILE)
    from common.config import PipelineConfig, save_pipeline_config
    from src.Data_Preprocessing import assign_splits, validate_splits

    cfg = cfg or PipelineConfig()
    rng = np.random.default_rng(seed)
    rows, n = [], 0
    for ident in range(1, n_identities + 1):
        for _ in range(int(rng.integers(images_range[0], images_range[1] + 1))):
            n += 1
            rows.append((f"{n:06d}.jpg", 1000 + ident))
    manifest = pd.DataFrame(rows, columns=["image", "identity"])
    splits = assign_splits(manifest, cfg)
    problems = validate_splits(splits, cfg)
    if problems:
        raise RuntimeError(problems)
    emb = make_synthetic_embeddings(splits, cfg.embedding_dim, noise, seed)

    paths.ensure()
    manifest.to_csv(paths.file(MANIFEST_FILE), index=False)
    splits.to_csv(paths.file(SPLITS_FILE), index=False)
    save_pipeline_config(cfg, paths)
    np.save(paths.file(EMBEDDINGS_FILE), emb)
    pd.DataFrame({"image": splits["image"], "row": np.arange(len(splits))}).to_csv(
        paths.file(EMBEDDINGS_INDEX_FILE), index=False)
    return splits, emb, cfg
