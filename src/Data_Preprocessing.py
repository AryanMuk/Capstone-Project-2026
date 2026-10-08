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

# =============================================================================
# SECTION 3/4 — CelebA crop and the upload ImageProcessor
# COMMIT: feat(preprocess): add CelebA crop and the upload ImageProcessor (validate/detect/crop/quality)
# =============================================================================


def crop_celeba(img: Image.Image, crop: int, size: int) -> Image.Image:
    """Centre-crop the 178x218 aligned CelebA image to crop x crop, resize to size x size.
    (crop size is a modelling assumption, see docs/PROJECT_LOG.md DEC-007; it is configurable.)"""
    w, h = img.size
    crop = min(crop, w, h)
    left, top = (w - crop) // 2, (h - crop) // 2
    return img.crop((left, top, left + crop, top + crop)).resize((size, size), Image.BILINEAR)


def load_celeba_array(path, cfg: PipelineConfig) -> np.ndarray:
    """File -> uint8 array (size, size, 3). Used by the offline embedding pipeline."""
    with Image.open(path) as im:
        return np.array(crop_celeba(im.convert("RGB"), cfg.celeba_crop, cfg.image_size))


@dataclass
class Stage:
    name: str
    ok: bool
    ms: float
    detail: str = ""


@dataclass
class ProcessResult:
    ok: bool
    reason: Optional[str]            # machine-readable rejection reason, None when ok
    message: str                     # human-readable explanation
    stages: List[Stage] = field(default_factory=list)
    face: Optional[np.ndarray] = None    # uint8 (size, size, 3) when ok
    box: Optional[List[float]] = None    # detected face box in the (possibly downscaled) image

    def stages_as_dicts(self) -> List[Dict]:
        return [asdict(s) for s in self.stages]

    def face_png_b64(self) -> Optional[str]:
        if self.face is None:
            return None
        buf = io.BytesIO()
        Image.fromarray(self.face).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")


def sharpness(face: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian of the grayscale face (higher = sharper)."""
    g = np.asarray(Image.fromarray(face).convert("L"), dtype=np.float64)
    lap = (-4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


class ImageProcessor:
    """Turns an arbitrary uploaded photo into a 160x160 face crop, or rejects it with a reason.

    Stages: validate -> detect (MTCNN) -> crop -> quality.
    Thresholds are conservative defaults that have NOT been tuned on real photos; they are
    constructor arguments so the team can adjust them after trying real images.
    """

    def __init__(self, image_size: int = 160, margin_frac: float = 0.15,
                 min_face_px: int = 60, min_sharpness: float = 20.0,
                 min_confidence: float = 0.9, max_bytes: int = 8_000_000,
                 min_image_px: int = 64, max_side_px: int = 1280, device: str = "cpu"):
        self.image_size, self.margin_frac = image_size, margin_frac
        self.min_face_px, self.min_sharpness = min_face_px, min_sharpness
        self.min_confidence, self.max_bytes = min_confidence, max_bytes
        self.min_image_px, self.max_side_px = min_image_px, max_side_px
        self.device = device
        self._mtcnn = None

    def _detector(self):
        if self._mtcnn is None:
            from facenet_pytorch import MTCNN   # lazy: heavy import, needs torch
            self._mtcnn = MTCNN(keep_all=True, device=self.device)
        return self._mtcnn

    def _crop(self, img: Image.Image, box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in box]
        side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * self.margin_frac)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        region = (int(round(cx - side / 2)), int(round(cy - side / 2)),
                  int(round(cx + side / 2)), int(round(cy + side / 2)))
        face = img.crop(region).resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.asarray(face)

    def process(self, data: bytes) -> ProcessResult:
        stages: List[Stage] = []

        def reject(reason: str, message: str, name: str, t0: float) -> ProcessResult:
            stages.append(Stage(name, False, (time.perf_counter() - t0) * 1000, message))
            return ProcessResult(False, reason, message, stages)

        # 1. validate ---------------------------------------------------------------
        t0 = time.perf_counter()
        if len(data) > self.max_bytes:
            return reject("image_too_large", f"file is larger than {self.max_bytes // 1_000_000} MB",
                          "validate", t0)
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception:  # noqa: BLE001 - any decode problem means an invalid upload
            return reject("invalid_image", "file is not a readable image", "validate", t0)
        if min(img.size) < self.min_image_px:
            return reject("image_too_small", f"image is smaller than {self.min_image_px}px",
                          "validate", t0)
        if max(img.size) > self.max_side_px:
            img.thumbnail((self.max_side_px, self.max_side_px))
        stages.append(Stage("validate", True, (time.perf_counter() - t0) * 1000,
                            f"{img.size[0]}x{img.size[1]}"))

        # 2. detect -----------------------------------------------------------------
        t0 = time.perf_counter()
        boxes, probs = self._detector().detect(img)
        if boxes is None:
            return reject("no_face", "no face detected", "detect", t0)
        keep = [i for i, p in enumerate(probs) if p is not None and p >= self.min_confidence]
        if not keep:
            return reject("no_face", "no confident face detected", "detect", t0)
        if len(keep) > 1:
            return reject("multiple_faces", f"{len(keep)} faces detected; exactly one is required",
                          "detect", t0)
        box = boxes[keep[0]]
        stages.append(Stage("detect", True, (time.perf_counter() - t0) * 1000,
                            f"confidence {float(probs[keep[0]]):.3f}"))

        # 3. crop -------------------------------------------------------------------
        t0 = time.perf_counter()
        face_px = float(min(box[2] - box[0], box[3] - box[1]))
        if face_px < self.min_face_px:
            return reject("face_too_small", f"face is {face_px:.0f}px, minimum is "
                          f"{self.min_face_px}px", "crop", t0)
        face = self._crop(img, box)
        stages.append(Stage("crop", True, (time.perf_counter() - t0) * 1000,
                            f"{self.image_size}x{self.image_size}, face {face_px:.0f}px"))

        # 4. quality ----------------------------------------------------------------
        t0 = time.perf_counter()
        sharp = sharpness(face)
        if sharp < self.min_sharpness:
            return reject("image_too_blurry", f"sharpness {sharp:.1f} below {self.min_sharpness}",
                          "quality", t0)
        stages.append(Stage("quality", True, (time.perf_counter() - t0) * 1000,
                            f"sharpness {sharp:.1f}"))
        return ProcessResult(True, None, "ok", stages, face, [float(v) for v in box])


# =============================================================================
# SECTION 4/4 — Write splits, config and report; command-line entry point
# COMMIT: feat(preprocess): write splits, config and report; add command-line entry point
# =============================================================================


def build_report(manifest_rows: int, eligible: pd.DataFrame, selected: pd.DataFrame,
                 splits: pd.DataFrame, cfg: PipelineConfig) -> Dict:
    return {
        "config": asdict(cfg),
        "manifest_images": int(manifest_rows),
        "eligible_identities": int(eligible.identity.nunique()),
        "selected_identities": int(selected.identity.nunique()),
        "selected_images": int(len(selected)),
        "known_identities": int(splits[splits.group == "known"].identity.nunique()),
        "unknown_identities": int(splits[splits.group == "unknown"].identity.nunique()),
        "counts": {f"{g}/{p}": int(n) for (g, p), n in
                   splits.groupby(["group", "partition"]).size().items()},
    }


def run(cfg: PipelineConfig, paths: Optional[Paths] = None, check_images: int = 20) -> Dict:
    paths = (paths or get_paths()).ensure()
    manifest = load_manifest(paths)
    eligible = filter_identities(manifest, cfg.min_images_per_identity)
    selected = select_subset(eligible, cfg.subset_identities, cfg.seed)
    splits = assign_splits(selected, cfg)
    problems = validate_splits(splits, cfg)
    if problems:
        raise RuntimeError("Invalid split: " + "; ".join(problems))

    if check_images:   # prove the crop works on real files before the expensive embedding step
        folder = images_dir(paths)
        for name in splits["image"].sample(min(check_images, len(splits)), random_state=cfg.seed):
            arr = load_celeba_array(folder / name, cfg)
            if arr.shape != (cfg.image_size, cfg.image_size, 3):
                raise RuntimeError(f"unexpected crop shape {arr.shape} for {name}")

    splits.to_csv(paths.file(SPLITS_FILE), index=False)
    save_pipeline_config(cfg, paths)
    report = build_report(len(manifest), eligible, selected, splits, cfg)
    write_json(paths.file("preprocess_report.json"), report)
    return report


def main(argv: Optional[list] = None) -> int:
    d = PipelineConfig()
    p = argparse.ArgumentParser(description="Filter identities and create known/unknown splits")
    p.add_argument("--subset", type=int, default=None,
                   help="number of identities to keep (omit for the full dataset)")
    p.add_argument("--min-images", type=int, default=d.min_images_per_identity)
    p.add_argument("--known-fraction", type=float, default=d.known_fraction)
    p.add_argument("--gallery", type=int, default=d.gallery_per_identity)
    p.add_argument("--val-fraction", type=float, default=d.val_fraction)
    p.add_argument("--crop", type=int, default=d.celeba_crop)
    p.add_argument("--cancelable-bits", type=int, default=d.cancelable_bits,
                   help="length of the cancelable code (multiple of 8); more bits = more accurate")
    p.add_argument("--seed", type=int, default=d.seed)
    args = p.parse_args(argv)

    cfg = PipelineConfig(seed=args.seed, subset_identities=args.subset,
                         min_images_per_identity=args.min_images,
                         known_fraction=args.known_fraction, gallery_per_identity=args.gallery,
                         val_fraction=args.val_fraction, celeba_crop=args.crop,
                         cancelable_bits=args.cancelable_bits)
    report = run(cfg)
    print(f"Identities: {report['selected_identities']} selected "
          f"({report['known_identities']} known, {report['unknown_identities']} unknown)")
    for key, n in report["counts"].items():
        print(f"  {key:<18} {n:>8} images")
    print(f"Wrote {get_paths().file(SPLITS_FILE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())