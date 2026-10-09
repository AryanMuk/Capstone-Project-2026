from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util
import os
import platform
import sys
from pathlib import Path
from typing import Dict, Optional

from common.artifacts import write_json
from common.config import DRIVE_MOUNT, IN_COLAB, drive_dir, get_paths
from common.crypto import KeyStore, MASTER_ENV


REQUIRED_PACKAGES = {  
    "numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn",
    "cryptography": "cryptography", "PIL": "Pillow",
    "flask": "Flask", "torch": "torch", "torchvision": "torchvision",
    "facenet_pytorch": "facenet-pytorch",
}


def _dist_version(dist: str) -> Optional[str]:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def inspect_environment() -> Dict:
    """Collect facts about the interpreter, packages and GPU. Imports torch lazily."""
    packages = {}
    for module, dist in REQUIRED_PACKAGES.items():
        ok = importlib.util.find_spec(module) is not None
        packages[dist] = {"importable": ok, "version": _dist_version(dist) if ok else None}
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "in_colab": IN_COLAB,
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "cuda_available": False,
        "gpu": None,
    }
    if packages["torch"]["importable"]:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    return info


def print_environment(info: Dict) -> None:
    print(f"Python {info['python']} on {info['platform']}  (Colab: {info['in_colab']})")
    print(f"CPU cores: {info['cpu_count']}   CUDA: {info['cuda_available']}   GPU: {info['gpu']}")
    for dist, d in info["packages"].items():
        mark = "ok " if d["importable"] else "MISSING"
        print(f"  [{mark}] {dist:<24} {d['version'] or ''}")
    if not IN_COLAB and not info["python"].startswith("3.11"):
        print("WARNING: the laptop requirements are pinned for Python 3.11 (see setup_env.ps1).")


def init_workspace() -> Dict:
    """Create work/artifacts/keys folders and make sure a master key exists."""
    paths = get_paths().ensure()
    had_env_key = MASTER_ENV in os.environ
    key_file = paths.keys / "master.key"
    existed = key_file.exists()
    ks = KeyStore.load_or_create(paths, create=True)
    info = {
        "work": str(paths.work), "raw": str(paths.raw), "artifacts": str(paths.artifacts),
        "keys": str(paths.keys),
        "master_key_source": "env" if had_env_key else ("existing file" if existed else "new file"),
        "master_key_fingerprint": ks.fingerprint(),  
    }
    write_json(paths.file("workspace_info.json"), info)
    return info


DEFAULT_ZIP_NAME = "celeba.zip"


def mount_drive() -> Path:
    """Mount Google Drive (Colab only) and return the project folder on Drive."""
    if not IN_COLAB:
        raise RuntimeError("mount_drive() only works inside Google Colab")
    from google.colab import drive  
    if not (DRIVE_MOUNT / "MyDrive").exists():
        drive.mount(str(DRIVE_MOUNT))
    return drive_dir()


def check_drive_inputs(zip_name: str = DEFAULT_ZIP_NAME) -> Path:
    """Confirm the dataset zip is where Data_Collection_Jaskaran expects it."""
    folder = mount_drive()
    zip_path = folder / zip_name
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Zip img_align_celeba/ together with identity_CelebA.txt, "
            f"name it {zip_name} and upload it to the 'capstone' folder in Google Drive.")
    print(f"Found {zip_path}  ({zip_path.stat().st_size / 1e9:.2f} GB)")
    return zip_path


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Project setup and environment checks")
    parser.add_argument("command", choices=["check", "init", "drive-check"])
    args = parser.parse_args(argv)

    if args.command == "check":
        info = inspect_environment()
        print_environment(info)
        missing = [d for d, v in info["packages"].items() if not v["importable"]]
        if missing:
            print(f"\nMissing packages: {', '.join(missing)}")
            return 1
        return 0
    if args.command == "init":
        info = init_workspace()
        for k, v in info.items():
            print(f"{k}: {v}")
        print("\nKeep keys/master.key private and never commit it. Copy it to the laptop "
              "yourself (or set CAPSTONE_MASTER_KEY_HEX) to run the demo.")
        return 0
    check_drive_inputs()
    return 0


if __name__ == "__main__":
    sys.exit(main())
