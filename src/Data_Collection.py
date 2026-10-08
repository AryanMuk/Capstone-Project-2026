import time
import zipfile
import shutil
import os

from pathlib import Path
from typing import Optional, Tuple

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
            zf.extract(m, target)  # ZipFile.extract strips absolute paths and '..'
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
        from src.Project_Setup_Suvrat import check_drive_inputs
        zip_path = check_drive_inputs(DEFAULT_ZIP_NAME)
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found")
    if IN_COLAB and str(zip_path).startswith("/content/drive"):
        local_zip = paths.work / zip_path.name       # Drive reads are slow; copy once
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


