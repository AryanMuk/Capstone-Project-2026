from __future__ import annotations

import argparse
import base64
import io
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from common.artifacts import SPLIT_COLUMNS, SPLITS_FILE, images_dir, load_manifest, write_json
from common.config import PipelineConfig, Paths, get_paths, save_pipeline_config

# =============================================================================
# SECTION 1/4 — Filter identities and select a reproducible subset
# COMMIT: feat(preprocess): filter identities by image count and select a reproducible subset
# =============================================================================


def filter_identities(manifest: pd.DataFrame, min_images: int) -> pd.DataFrame:
    """Keep only identities that have at least `min_images` images."""
    counts = manifest.groupby("identity")["image"].transform("size")
    return manifest.loc[counts >= min_images].reset_index(drop=True)


def select_subset(df: pd.DataFrame, subset_identities: Optional[int], seed: int) -> pd.DataFrame:
    """Random subset of identities (None = keep all). Same seed -> same subset."""
    ids = np.array(sorted(df["identity"].unique()))
    if subset_identities is None or subset_identities >= len(ids):
        return df
    chosen = np.random.default_rng(seed).choice(ids, size=subset_identities, replace=False)
    return df.loc[df["identity"].isin(chosen)].reset_index(drop=True)


# =============================================================================
# SECTION 2/4 — Known/unknown identity split with gallery/val/test roles
# COMMIT: feat(preprocess): add known/unknown identity split with gallery/val/test roles
# =============================================================================


def assign_splits(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Return columns image, identity, group (known|unknown), role (gallery|probe),
    partition (gallery|val|test), sorted by image name for sequential disk reads."""
    rng = np.random.default_rng(cfg.seed)
    identities = np.array(sorted(df["identity"].unique()))
    shuffled = rng.permutation(identities)
    n_known = int(round(len(identities) * cfg.known_fraction))
    n_known = min(max(n_known, 1), len(identities) - 1)
    known_ids, unknown_ids = shuffled[:n_known], shuffled[n_known:]
    if len(unknown_ids) < 2:
        raise ValueError("Need at least 2 unknown identities for separate val/test sets; "
                         "use more identities (larger --subset or lower --known-fraction).")

    by_identity = {k: g["image"].to_numpy() for k, g in df.groupby("identity")}
    rows = []
    for ident in known_ids:
        imgs = rng.permutation(by_identity[ident])
        gallery, probes = imgs[:cfg.gallery_per_identity], imgs[cfg.gallery_per_identity:]
        n_val = min(int(round(len(probes) * cfg.val_fraction)), max(len(probes) - 1, 0))
        rows += [(i, ident, "known", "gallery", "gallery") for i in gallery]
        rows += [(i, ident, "known", "probe", "val") for i in probes[:n_val]]
        rows += [(i, ident, "known", "probe", "test") for i in probes[n_val:]]

    n_unknown_val = min(max(int(round(len(unknown_ids) * cfg.val_fraction)), 1),
                        len(unknown_ids) - 1)
    for pos, ident in enumerate(unknown_ids):
        part = "val" if pos < n_unknown_val else "test"
        rows += [(i, ident, "unknown", "probe", part) for i in by_identity[ident]]

    return (pd.DataFrame(rows, columns=SPLIT_COLUMNS)
            .sort_values("image").reset_index(drop=True))


def validate_splits(splits: pd.DataFrame, cfg: PipelineConfig) -> List[str]:
    """Return a list of problems (empty = split is sound)."""
    problems = []
    if splits["image"].duplicated().any():
        problems.append("an image appears in more than one row")
    known, unknown = splits[splits.group == "known"], splits[splits.group == "unknown"]
    if set(known.identity) & set(unknown.identity):
        problems.append("an identity is both known and unknown")
    gal_counts = known[known.role == "gallery"].groupby("identity").size()
    if len(gal_counts) != known.identity.nunique() or (gal_counts != cfg.gallery_per_identity).any():
        problems.append("a known identity does not have exactly gallery_per_identity gallery images")
    if (unknown.role != "probe").any():
        problems.append("unknown identities must contain probes only")
    if set(unknown[unknown.partition == "val"].identity) & set(
            unknown[unknown.partition == "test"].identity):
        problems.append("unknown identities leak between val and test")
    for part in ("val", "test"):
        if known[known.partition == part].empty or unknown[unknown.partition == part].empty:
            problems.append(f"partition '{part}' has no known or no unknown probes")
    return problems