from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common import cancelable, db
from common.artifacts import load_embeddings, user_id_for
from common.config import SCHEMES, PipelineConfig, get_paths, load_pipeline_config
from common.crypto import KeyStore, TemplateIntegrityError
from common.schemes import FLOAT_SCHEMES, open_blob

@dataclass
class GalleryMatcher:
    scheme: str
    user_ids: List[str]
    gallery: np.ndarray                    
    signs: Optional[np.ndarray] = None     
    proj: Optional[np.ndarray] = None      
    rejected: List[Tuple[str, str]] = field(default_factory=list)  
    index: Dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {u: i for i, u in enumerate(self.user_ids)}

    def __len__(self) -> int:
        return len(self.user_ids)


def _build_matcher(scheme: str, ks: KeyStore, rows, dim: int, bits: int) -> GalleryMatcher:
    ids, mats, signs, rejected = [], [], [], []
    for uid, version, blob in rows:
        try:
            mats.append(open_blob(scheme, ks, uid, version, blob, dim, bits))
        except TemplateIntegrityError as exc:    
            rejected.append((uid, str(exc)))
            continue
        ids.append(uid)
        if scheme == "cancelable":
            signs.append(ks.cancelable_signs(uid, version, dim))
    width = bits if scheme == "cancelable" else dim
    dtype = np.int8 if scheme == "cancelable" else np.float32
    gallery = np.stack(mats) if mats else np.zeros((0, width), dtype=dtype)
    return GalleryMatcher(
        scheme=scheme, user_ids=ids, gallery=gallery,
        signs=(np.stack(signs) if signs else np.zeros((0, dim), np.int8)) if scheme == "cancelable"
        else None,
        proj=ks.cancelable_projection(bits, dim) if scheme == "cancelable" else None,
        rejected=rejected)


def open_gallery(conn: sqlite3.Connection, scheme: str, ks: KeyStore, dim: int,
                 bits: int) -> GalleryMatcher:
    """Decrypt every active template of `scheme` (the 1:N gallery)."""
    return _build_matcher(scheme, ks, db.iter_active_templates(conn, scheme), dim, bits)


def open_user(conn: sqlite3.Connection, scheme: str, ks: KeyStore, user_id: str, dim: int,
              bits: int) -> GalleryMatcher:
    """Decrypt a single user's template (the 1:1 case). Unknown user -> KeyError;
    tampered template -> TemplateIntegrityError."""
    found = db.get_active_template(conn, user_id, scheme)
    if found is None:
        raise KeyError(user_id)
    version, blob = found
    matcher = _build_matcher(scheme, ks, [(user_id, version, blob)], dim, bits)
    if matcher.rejected:
        raise TemplateIntegrityError(matcher.rejected[0][1])
    return matcher

def _device(device: Optional[str]) -> str:
    import torch
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _cancelable_agreement(probes: np.ndarray, signs: np.ndarray, proj: np.ndarray,
                          codes: np.ndarray, pairwise: bool, batch: int,
                          device: Optional[str]) -> np.ndarray:
    """Mean code agreement in [-1, 1].
    pairwise=False: every probe against every user -> (B, U) (1:N)
    pairwise=True : probe i against user i only      -> (B,)  (1:1; signs/codes already gathered)"""
    import torch
    dev = _device(device)
    P = torch.as_tensor(proj, device=dev)
    S = torch.as_tensor(signs, device=dev, dtype=torch.float32)
    C = torch.as_tensor(codes, device=dev, dtype=torch.float32)
    out = []
    for i in range(0, len(probes), batch):
        x = torch.as_tensor(probes[i:i + batch], device=dev)
        if pairwise:
            z = (x * S[i:i + batch]) @ P.T                               
            out.append((torch.where(z >= 0, 1.0, -1.0) * C[i:i + batch]).mean(-1).cpu().numpy())
        else:
            z = (x[:, None, :] * S[None, :, :]) @ P.T                    
            out.append((torch.where(z >= 0, 1.0, -1.0) * C[None, :, :]).mean(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,) if pairwise else (0, len(S)))


def score_matrix(matcher: GalleryMatcher, probes: np.ndarray, batch: Optional[int] = None,
                 device: Optional[str] = None) -> np.ndarray:
    """(B, U) score of every probe against every enrolled user (higher = more similar)."""
    probes = np.asarray(probes, dtype=np.float32)
    if probes.ndim == 1:
        probes = probes[None]
    if matcher.scheme in FLOAT_SCHEMES:
        return probes @ matcher.gallery.T
    if len(matcher) == 0:
        return np.zeros((len(probes), 0), dtype=np.float32)
    width = max(matcher.gallery.shape[1], matcher.signs.shape[1])
    batch = batch or int(max(1, min(256, 6e7 // (len(matcher) * width))))
    agree = _cancelable_agreement(probes, matcher.signs, matcher.proj, matcher.gallery,
                                  pairwise=False, batch=batch, device=device)
    return cancelable.score_from_agreement(agree).astype(np.float32)


def score_pairs(matcher: GalleryMatcher, probes: np.ndarray, user_idx: np.ndarray,
                batch: int = 4096, device: Optional[str] = None) -> np.ndarray:
    """(B,) score of probe i against user user_idx[i]; this is the 1:1 claim check."""
    probes = np.asarray(probes, dtype=np.float32)
    if probes.ndim == 1:
        probes = probes[None]
    user_idx = np.asarray(user_idx)
    if matcher.scheme in FLOAT_SCHEMES:
        return np.einsum("ij,ij->i", probes, matcher.gallery[user_idx])
    agree = _cancelable_agreement(probes, matcher.signs[user_idx], matcher.proj,
                                  matcher.gallery[user_idx], pairwise=True, batch=batch,
                                  device=device)
    return cancelable.score_from_agreement(agree).astype(np.float32)

@dataclass
class Decision:
    accepted: bool
    reason: str                              
    score: Optional[float]
    threshold: float
    user_id: Optional[str] = None            
    top: List[Tuple[str, float]] = field(default_factory=list)


def verify(matcher: GalleryMatcher, probe: np.ndarray, claimed_user: str,
           threshold: float) -> Decision:
    """1:1 - does `probe` belong to `claimed_user`?"""
    idx = matcher.index.get(claimed_user)
    if idx is None:
        return Decision(False, "unknown_user", None, threshold, claimed_user)
    score = float(score_pairs(matcher, probe, np.array([idx]))[0])
    ok = score >= threshold
    return Decision(ok, "match" if ok else "below_threshold", score, threshold, claimed_user,
                    [(claimed_user, score)])


def identify(matcher: GalleryMatcher, probe: np.ndarray, threshold: float,
             top_k: int = 5) -> Decision:
    """1:N - who is this? Open-set: the best match is accepted only if it clears `threshold`,
    otherwise the person is reported as not enrolled."""
    if len(matcher) == 0:
        return Decision(False, "empty_gallery", None, threshold)
    scores = score_matrix(matcher, probe)[0]
    order = np.argsort(-scores)[:top_k]
    top = [(matcher.user_ids[i], float(scores[i])) for i in order]
    best_uid, best = top[0]
    ok = best >= threshold
    return Decision(ok, "match" if ok else "below_threshold", best,
                    threshold, best_uid if ok else None, top)


def check_request_freshness(conn: sqlite3.Connection, nonce: str, issued_at: float,
                            now: Optional[float] = None,
                            window_s: float = 60.0) -> Tuple[bool, str]:
    """Replay protection: a request must be recent and carry a nonce never seen before.
    Returns (ok, reason) with reason in {ok, stale_request, replayed_nonce}."""
    now = time.time() if now is None else now
    if not math.isfinite(issued_at) or abs(now - issued_at) > window_s:
        return False, "stale_request"
    if not db.register_nonce(conn, nonce):
        return False, "replayed_nonce"
    return True, "ok"

def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Demonstrate verification/identification on test probes")
    p.add_argument("--scheme", choices=SCHEMES, default="aes")
    p.add_argument("--n", type=int, default=5, help="probes per kind (known / unknown)")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args(argv)

    paths, cfg = get_paths(), load_pipeline_config()
    from src.Embedding_Encryption import TEMPLATES_DB
    emb, splits = load_embeddings(paths)
    conn = db.connect(paths.file(TEMPLATES_DB))
    ks = KeyStore.load_or_create(paths, create=False)
    matcher = open_gallery(conn, args.scheme, ks, cfg.embedding_dim, cfg.cancelable_bits)
    print(f"Opened {len(matcher)} {args.scheme} templates "
          f"({len(matcher.rejected)} rejected as tampered)")

    test = splits[splits.partition == "test"]
    for label, group in (("KNOWN", "known"), ("UNKNOWN", "unknown")):
        for _, row in test[test.group == group].head(args.n).iterrows():
            d = identify(matcher, emb[row["row"]], args.threshold)
            truth = user_id_for(row["identity"])
            print(f"[{label}] truth={truth}  -> accepted={d.accepted} best={d.top[0][0]} "
                  f"score={d.score:.3f} ({d.reason})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
