from __future__ import annotations
import argparse
import sys
import time
import zipfile
import shutil
import os
import random
import pandas as pd 

from pathlib import Path
from typing import Optional, Tuple, Dict
from PIL import Image

from common.artifacts import MANIFEST_FILE
from common.config import IN_COLAB

EXPECTED_IMAGES = 202_599
EXPECTED_IDENTITIES = 10_177
IDENTITY_FILE_NAME = "identity_CelebA.txt"
DEFAULT_ZIP_NAME = "celeba.zip"


def _marker_name(zip_path: Path) -> str:
    return f".extracted_{zip_path.name}_{zip_path.stat().st_size}"


def extract_zip(zip_path: Path, target: Path) -> Path:
    """Extract once; a marker file records that this exact zip was already extracted."""
    target.mkdir(parents=True, exist_ok=True)
    marker = target / _marker_name(zip_path)
    if marker.exists():
        print(f"Already extracted: {zip_path.name} -> {target}")
        return target
    start = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for i, m in enumerate(members, 1):
            zf.extract(m, target)
            if i % 25_000 == 0:
                print(f"  extracted {i}/{len(members)} files")
    marker.write_text(f"{time.time() - start:.1f}s")
    print(f"Extracted {zip_path.name} in {time.time() - start:.0f}s")
    return target


def stage_dataset(zip_path: Optional[Path], dir_path: Optional[Path], paths: Paths) -> Path:
    """Return the folder to search for the images and identity file."""
    if dir_path is not None:
        if not dir_path.is_dir():
            raise FileNotFoundError(f"--dir {dir_path} is not a folder")
        return dir_path
    if zip_path is None:
        if not IN_COLAB:
            raise ValueError("Pass --zip <path to celeba zip> or --dir <extracted folder>.")
        from src.Project_Setup import check_drive_inputs
        zip_path = check_drive_inputs(DEFAULT_ZIP_NAME)
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found")
    if IN_COLAB and str(zip_path).startswith("/content/drive"):
        local_zip = paths.work / zip_path.name
        if not local_zip.exists() or local_zip.stat().st_size != zip_path.stat().st_size:
            print(f"Copying {zip_path.name} from Drive to local disk ...")
            shutil.copy2(zip_path, local_zip)
        zip_path = local_zip
    return extract_zip(zip_path, paths.raw)

def _dir_with_most_jpgs(root: Path) -> Path:
    best, best_n = None, -1
    for folder, _dirs, files in os.walk(root):
        n = sum(1 for f in files if f.lower().endswith(".jpg"))
        if n > best_n:
            best, best_n = Path(folder), n
    if best is None or best_n <= 0:
        raise FileNotFoundError(f"No .jpg files found under {root}")
    return best


def locate_celeba(root: Path) -> Tuple[Path, Path]:
    """Find (images_dir, identity_file) wherever they sit inside `root`."""
    identity_file = next(root.rglob(IDENTITY_FILE_NAME), None)
    if identity_file is None:
        raise FileNotFoundError(
            f"{IDENTITY_FILE_NAME} not found under {root}. It must be inside the zip/folder.")
    first = next(root.rglob("000001.jpg"), None)
    images_dir = first.parent if first is not None else _dir_with_most_jpgs(root)
    return images_dir, identity_file


def parse_identity_file(path: Path) -> pd.DataFrame:
    """identity_CelebA.txt has lines like '000001.jpg 2880'."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["image", "identity"],
                     dtype={"image": str, "identity": int})
    if df["image"].duplicated().any():
        raise ValueError(f"{path} lists some images more than once")
    if not df["image"].str.lower().str.endswith(".jpg").all():
        raise ValueError(f"{path} contains entries that are not .jpg files")
    return df

def verify_images(df: pd.DataFrame, images_dir: Path, sample: int = 500,
                  seed: int = 0) -> Tuple[pd.DataFrame, Dict]:
    """Keep rows whose file exists; open a random sample to catch corrupt files."""
    on_disk = set(os.listdir(images_dir))
    present_mask = df["image"].isin(on_disk)
    missing = df.loc[~present_mask, "image"].tolist()
    present = df.loc[present_mask].reset_index(drop=True)

    names = present["image"].tolist()
    picked = random.Random(seed).sample(names, min(sample, len(names)))
    corrupt, sizes = [], {}
    for name in picked:
        try:
            with Image.open(images_dir / name) as im:
                im.verify()
            with Image.open(images_dir / name) as im:   
                sizes[im.size] = sizes.get(im.size, 0) + 1
        except Exception as exc:
            corrupt.append({"image": name, "error": str(exc)})
    return present, {
        "n_missing": len(missing), "missing_examples": missing[:10],
        "n_extra_files_on_disk": len(on_disk - set(df["image"])),
        "sample_checked": len(picked), "corrupt": corrupt,
        "sample_image_sizes": {f"{w}x{h}": n for (w, h), n in sizes.items()},
    }


def build_report(present: pd.DataFrame, checks: Dict, images_dir: Path,
                 identity_file: Path) -> Dict:
    per_id = present.groupby("identity").size()
    warnings = []
    if len(present) != EXPECTED_IMAGES:
        warnings.append(f"{len(present)} images found, published CelebA has {EXPECTED_IMAGES}")
    if per_id.size != EXPECTED_IDENTITIES:
        warnings.append(f"{per_id.size} identities found, published CelebA has "
                        f"{EXPECTED_IDENTITIES}")
    if checks["n_missing"]:
        warnings.append(f"{checks['n_missing']} images listed in the identity file are missing")
    if checks["corrupt"]:
        warnings.append(f"{len(checks['corrupt'])} sampled images failed to decode")
    return {
        "images_dir": str(images_dir), "identity_file": str(identity_file),
        "n_images": int(len(present)), "n_identities": int(per_id.size),
        "images_per_identity": {
            "min": int(per_id.min()), "median": float(per_id.median()),
            "mean": round(float(per_id.mean()), 2), "max": int(per_id.max()),
            "identities_with_10_or_more": int((per_id >= 10).sum()),
        },
        "checks": checks, "warnings": warnings,
    }


def collect(zip_path: Optional[Path], dir_path: Optional[Path], sample: int,
            paths: Optional[Paths] = None) -> Dict:
    paths = (paths or get_paths()).ensure()
    root = stage_dataset(zip_path, dir_path, paths)
    images_dir, identity_file = locate_celeba(root)
    df = parse_identity_file(identity_file)
    present, checks = verify_images(df, images_dir, sample=sample)
    if present.empty:
        raise RuntimeError(f"None of the images listed in {identity_file} exist in {images_dir}")
    present.to_csv(paths.file(MANIFEST_FILE), index=False)
    write_dataset_location(paths, images_dir, identity_file)
    report = build_report(present, checks, images_dir, identity_file)
    write_json(paths.file("data_report.json"), report)
    return report

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage, locate and verify the CelebA dataset")
    parser.add_argument("--zip", type=Path, help="zip with images + identity_CelebA.txt "
                        f"(Colab default: {drive_dir() / DEFAULT_ZIP_NAME})")
    parser.add_argument("--dir", type=Path, help="already-extracted folder (skips extraction)")
    parser.add_argument("--sample", type=int, default=500, help="images to decode-check")
    args = parser.parse_args(argv)

    report = collect(args.zip, args.dir, args.sample)
    print(f"\nImages: {report['n_images']}   Identities: {report['n_identities']}")
    print(f"Images per identity: {report['images_per_identity']}")
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    print(f"Manifest written to {get_paths().file(MANIFEST_FILE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())