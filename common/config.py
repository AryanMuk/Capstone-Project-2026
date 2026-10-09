from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMES = ("plaintext", "aes", "ecies", "cancelable")


def _detect_colab() -> bool:
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return True
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        return False


IN_COLAB = _detect_colab()
DRIVE_MOUNT = Path("/content/drive")


def drive_dir() -> Path:
    """Folder on Google Drive that holds the dataset zip and persisted artifacts."""
    env = os.environ.get("CAPSTONE_DRIVE_DIR")
    return Path(env) if env else DRIVE_MOUNT / "MyDrive" / "capstone"


@dataclass(frozen=True)
class Paths:
    work: Path        
    raw: Path         
    artifacts: Path   
    keys: Path        

    def file(self, name: str) -> Path:
        return self.artifacts / name

    def ensure(self) -> "Paths":
        for p in (self.work, self.raw, self.artifacts, self.keys):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths() -> Paths:
    work = Path(os.environ.get("CAPSTONE_WORK_DIR")
                or ("/content/work" if IN_COLAB else REPO_ROOT / "work"))
    art_env = os.environ.get("CAPSTONE_ARTIFACTS_DIR")
    if art_env:
        artifacts = Path(art_env)
    elif IN_COLAB and (DRIVE_MOUNT / "MyDrive").exists():
        artifacts = drive_dir() / "artifacts"
    else:
        artifacts = work / "artifacts"
    return Paths(work=work, raw=work / "raw", artifacts=artifacts, keys=artifacts / "keys")


@dataclass
class PipelineConfig:
    """Parameters that must stay identical across all pipeline stages."""
    seed: int = 42
    subset_identities: Optional[int] = None   
    min_images_per_identity: int = 10
    known_fraction: float = 0.8               
    gallery_per_identity: int = 5             
    val_fraction: float = 0.5                 
    image_size: int = 160                     
    celeba_crop: int = 148                    
    embedding_dim: int = 512
    cancelable_bits: int = 256                
    imp_per_probe: int = 10                   
    batch_size: int = 256

    def __post_init__(self) -> None:
        if self.cancelable_bits % 8 != 0:
            raise ValueError("cancelable_bits must be a multiple of 8")
        if not 0.0 < self.known_fraction < 1.0:
            raise ValueError("known_fraction must be in (0, 1)")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        if self.gallery_per_identity < 1:
            raise ValueError("gallery_per_identity must be >= 1")
        if self.min_images_per_identity <= self.gallery_per_identity:
            raise ValueError("min_images_per_identity must exceed gallery_per_identity "
                             "so every known identity keeps at least one probe")


CONFIG_FILE = "pipeline_config.json"


def save_pipeline_config(cfg: PipelineConfig, paths: Optional[Paths] = None) -> Path:
    paths = (paths or get_paths()).ensure()
    out = paths.file(CONFIG_FILE)
    out.write_text(json.dumps(asdict(cfg), indent=2))
    return out


def load_pipeline_config(paths: Optional[Paths] = None) -> PipelineConfig:
    """Config written by Data_Preprocessing; defaults if it has not been written yet."""
    paths = paths or get_paths()
    f = paths.file(CONFIG_FILE)
    if not f.exists():
        return PipelineConfig()
    known = {x.name for x in fields(PipelineConfig)}
    data = {k: v for k, v in json.loads(f.read_text()).items() if k in known}
    return PipelineConfig(**data)
