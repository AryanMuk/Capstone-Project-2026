from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from common.config import Paths, get_paths

LOCATION_FILE = "dataset_location.json"
MANIFEST_FILE = "manifest.csv"
SPLITS_FILE = "splits.csv"
EMBEDDINGS_FILE = "embeddings.npy"
EMBEDDINGS_INDEX_FILE = "embeddings_index.csv"
SPLIT_COLUMNS = ["image", "identity", "group", "role", "partition"]


def user_id_for(identity: int) -> str:
    """Stable user id for a CelebA identity number."""
    return f"id_{int(identity):05d}"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Cannot serialise {type(o)}")


def write_dataset_location(paths: Paths, images_dir: Path, identity_file: Path) -> None:
    write_json(paths.file(LOCATION_FILE),
               {"images_dir": str(images_dir), "identity_file": str(identity_file)})


def images_dir(paths: Optional[Paths] = None) -> Path:
    """Folder containing the CelebA jpgs. CAPSTONE_IMAGES_DIR overrides (e.g. on the laptop)."""
    import os
    env = os.environ.get("CAPSTONE_IMAGES_DIR")
    if env:
        return Path(env)
    paths = paths or get_paths()
    loc = paths.file(LOCATION_FILE)
    if not loc.exists():
        raise FileNotFoundError(
            f"{loc} not found. Run Data_Collection first, or set CAPSTONE_IMAGES_DIR.")
    return Path(read_json(loc)["images_dir"])


def load_manifest(paths: Optional[Paths] = None) -> pd.DataFrame:
    paths = paths or get_paths()
    f = paths.file(MANIFEST_FILE)
    if not f.exists():
        raise FileNotFoundError(f"{f} not found. Run Data_Collection first.")
    return pd.read_csv(f, dtype={"image": str, "identity": int})


def load_splits(paths: Optional[Paths] = None) -> pd.DataFrame:
    paths = paths or get_paths()
    f = paths.file(SPLITS_FILE)
    if not f.exists():
        raise FileNotFoundError(f"{f} not found. Run Data_Preprocessing first.")
    df = pd.read_csv(f, dtype={"image": str, "identity": int})
    missing = set(SPLIT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{f} is missing columns {sorted(missing)}")
    return df


def load_embeddings(paths: Optional[Paths] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """Returns (embeddings, splits) where splits gains a `row` column indexing embeddings."""
    paths = paths or get_paths()
    emb_f, idx_f = paths.file(EMBEDDINGS_FILE), paths.file(EMBEDDINGS_INDEX_FILE)
    for f in (emb_f, idx_f):
        if not f.exists():
            raise FileNotFoundError(f"{f} not found. Run Embedding_Generation first.")
    emb = np.load(emb_f)
    idx = pd.read_csv(idx_f, dtype={"image": str, "row": int})
    splits = load_splits(paths)
    merged = splits.merge(idx, on="image", how="inner", validate="one_to_one")
    if len(merged) != len(splits):
        raise ValueError(f"{len(splits) - len(merged)} split images have no embedding")
    if merged["row"].max() >= len(emb):
        raise ValueError("embeddings_index.csv refers to rows beyond embeddings.npy")
    return emb, merged
