from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from common import cancelable, db
from common.artifacts import load_embeddings, user_id_for, write_json
from common.config import SCHEMES, PipelineConfig, Paths, get_paths, load_pipeline_config
from common.crypto import KeyStore, TemplateIntegrityError
from common.schemes import open_blob, protect

TEMPLATES_DB = "templates.sqlite"

def build_user_templates(emb: np.ndarray, splits: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """Mean of each known identity's gallery embeddings, re-normalised to unit length.
    `splits` must carry the `row` column added by common.artifacts.load_embeddings."""
    gallery = splits[(splits["group"] == "known") & (splits["role"] == "gallery")]
    if gallery.empty:
        raise ValueError("no gallery images in splits")
    user_ids, vectors = [], []
    for identity, g in gallery.groupby("identity"):
        v = emb[g["row"].to_numpy()].mean(axis=0)
        n = np.linalg.norm(v)
        if n == 0:
            raise ValueError(f"identity {identity} has a zero-length mean template")
        user_ids.append(user_id_for(identity))
        vectors.append((v / n).astype(np.float32))
    return user_ids, np.stack(vectors)


def enroll_users(conn: sqlite3.Connection, ks: KeyStore, user_ids: Sequence[str],
                 vectors: np.ndarray, bits: int, schemes: Sequence[str] = SCHEMES,
                 version: int = 1, source: str = "celeba",
                 names: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, float]]:
    """Protect every vector under each scheme and store it. Returns per-scheme timings of the
    protect() call only (database writes are excluded so schemes compare fairly)."""
    if len(user_ids) != len(vectors):
        raise ValueError("user_ids and vectors differ in length")
    for uid in user_ids:
        db.upsert_user(conn, uid, (names or {}).get(uid, uid), source=source, key_version=version)
    timings: Dict[str, Dict[str, float]] = {}
    for scheme in schemes:
        spent = 0.0
        for uid, vec in zip(user_ids, vectors):
            t0 = time.perf_counter()
            blob = protect(scheme, ks, uid, version, vec, bits)
            spent += time.perf_counter() - t0
            db.put_template(conn, uid, scheme, version, blob)
        timings[scheme] = {"count": len(user_ids), "enroll_ms_total": spent * 1000,
                           "enroll_ms_mean": spent * 1000 / max(len(user_ids), 1)}
    conn.commit()
    return timings


def reissue_user(conn: sqlite3.Connection, ks: KeyStore, user_id: str, vector: np.ndarray,
                 bits: int, schemes: Sequence[str] = SCHEMES) -> int:
    """Revocation: move the user to key_version + 1, enrol `vector` (a FRESH capture of the
    user) under every scheme with the new keys, then delete all older versions.
    Returns the new key version. The old blobs, and codes made with the old keys, stop working."""
    user = db.get_user(conn, user_id)
    if user is None:
        raise KeyError(user_id)
    new_version = db.bump_key_version(conn, user_id)
    enroll_users(conn, ks, [user_id], np.asarray(vector, dtype=np.float32)[None], bits,
                 schemes, version=new_version, source=user["source"],
                 names={user_id: user["display_name"]})
    db.purge_old_versions(conn, user_id)
    conn.commit()
    return new_version


def verify_templates(conn: sqlite3.Connection, ks: KeyStore, user_ids: Sequence[str],
                     vectors: np.ndarray, dim: int, bits: int, sample: int = 200,
                     seed: int = 0) -> Dict[str, Dict]:
    """Decrypt a sample of stored templates and compare with what was enrolled; flip one byte
    of each sampled blob and confirm protected schemes refuse it. Raises on any failure."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(user_ids), size=min(sample, len(user_ids)), replace=False)
    report: Dict[str, Dict] = {}
    for scheme in SCHEMES:
        max_err, tamper_detected = 0.0, 0
        for i in picks:
            uid = user_ids[i]
            version, blob = db.get_active_template(conn, uid, scheme)
            opened = open_blob(scheme, ks, uid, version, blob, dim, bits)
            if scheme == "cancelable":
                expected = cancelable.binarize(
                    vectors[i], ks.cancelable_signs(uid, version, dim),
                    ks.cancelable_projection(bits, dim))
                err = float(np.abs(opened.astype(np.int16) - expected.astype(np.int16)).max())
            else:
                err = float(np.abs(opened - vectors[i]).max())
            max_err = max(max_err, err)
            bad = bytearray(blob)
            bad[len(bad) // 2] ^= 0x01
            try:
                open_blob(scheme, ks, uid, version, bytes(bad), dim, bits)
            except TemplateIntegrityError:
                tamper_detected += 1
        report[scheme] = {"checked": len(picks), "max_roundtrip_error": max_err,
                          "tamper_detected": tamper_detected}
        if max_err != 0.0:
            raise RuntimeError(f"{scheme}: stored template does not round-trip (err {max_err})")
        if scheme != "plaintext" and tamper_detected != len(picks):
            raise RuntimeError(f"{scheme}: tampering went undetected in "
                               f"{len(picks) - tamper_detected} cases")
    return report


def build_database(cfg: PipelineConfig, paths: Optional[Paths] = None,
                   db_path: Optional[Path] = None) -> Dict:
    paths = (paths or get_paths()).ensure()
    db_path = Path(db_path) if db_path else paths.file(TEMPLATES_DB)
    emb, splits = load_embeddings(paths)
    user_ids, vectors = build_user_templates(emb, splits)
    ks = KeyStore.load_or_create(paths)

    if db_path.exists():
        db_path.unlink()         
    conn = db.connect(db_path)
    db.init_schema(conn)
    timings = enroll_users(conn, ks, user_ids, vectors, cfg.cancelable_bits)
    checks = verify_templates(conn, ks, user_ids, vectors, cfg.embedding_dim, cfg.cancelable_bits)
    stats = db.template_stats(conn)
    conn.close()
    report = {"db": str(db_path), "n_users": len(user_ids), "key_fingerprint": ks.fingerprint(),
              "timings": timings, "verification": checks, "storage": stats}
    write_json(paths.file("enroll_timings.json"), timings)
    write_json(paths.file("encryption_report.json"), report)
    return report


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Enrol all known users under every scheme")
    p.add_argument("--db", type=Path, default=None, help=f"output database (default {TEMPLATES_DB})")
    args = p.parse_args(argv)
    report = build_database(load_pipeline_config(), db_path=args.db)
    print(f"Enrolled {report['n_users']} users into {report['db']}  "
          f"(key fingerprint {report['key_fingerprint']})")
    print(f"{'scheme':<11}{'enrol ms/template':>20}{'bytes/template':>18}")
    for s in SCHEMES:
        print(f"{s:<11}{report['timings'][s]['enroll_ms_mean']:>20.3f}"
              f"{report['storage'][s]['mean_bytes']:>18.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
