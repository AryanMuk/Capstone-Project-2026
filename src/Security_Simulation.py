from __future__ import annotations

import argparse
import sys
import time
import uuid
import zlib
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from common import cancelable, db
from common.artifacts import write_json
from common.config import SCHEMES, get_paths
from common.crypto import TemplateIntegrityError, aes_gcm_decrypt, ecies_decrypt
from common.schemes import aad_for, protect
from src.Embedding_Encryption import enroll_users, reissue_user
from src.Query_Authorisation import (check_request_freshness, open_gallery, open_user,
                                              score_pairs, verify)
from src.Threat_Scenario_Setup import (SCENARIOS, Lab, UserKeyMaterial, db_only_view,
                                             leak_user_keys, master_key_view)

TRIALS_PER_VICTIM = 10 


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity."""
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


class Defender:
    """The victim system as the attacker sees it from outside: open gallery + thresholds."""

    def __init__(self, lab: Lab):
        self.lab = lab
        conn = lab.connect()
        self.matchers = {s: open_gallery(conn, s, lab.ks, lab.cfg.embedding_dim,
                                         lab.cfg.cancelable_bits) for s in SCHEMES}
        conn.close()
        m = self.matchers["plaintext"]
        self.idx = np.array([m.index[u] for u in lab.victims])

    def asr(self, scheme: str, forged: np.ndarray, rows: Optional[np.ndarray] = None) -> float:
        """Share of victims accepted. `forged` has one credential per victim (same order as
        lab.victims); `rows` optionally restricts the calculation to some victims."""
        rows = np.arange(len(self.idx)) if rows is None else np.asarray(rows)
        scores = score_pairs(self.matchers[scheme], forged[rows], self.idx[rows])
        return float((scores >= self.lab.thresholds[scheme]).mean())

    def zero_effort_asr(self, scheme: str, rng: np.random.Generator,
                        rows: Optional[np.ndarray] = None) -> float:
        """Attacker presents their own face for each victim, TRIALS_PER_VICTIM times."""
        victims = np.arange(len(self.idx)) if rows is None else np.asarray(rows)
        claims = np.repeat(victims, TRIALS_PER_VICTIM)
        faces = self.lab.attacker_probes[rng.integers(0, len(self.lab.attacker_probes),
                                                      len(claims))]
        scores = score_pairs(self.matchers[scheme], faces, self.idx[claims])
        return float((scores >= self.lab.thresholds[scheme]).mean())


def naive_parse(blob: bytes, dim: int) -> Optional[np.ndarray]:
    """What an attacker without keys can do: read the bytes as a float32 embedding.
    Returns the vector only if it looks like a genuine unit-length embedding."""
    if len(blob) < 4 * dim:
        return None
    vec = np.frombuffer(blob[:4 * dim], dtype="<f4")
    if not np.isfinite(vec).all() or abs(float(np.linalg.norm(vec.astype(np.float64))) - 1.0) > 1e-3:
        return None
    return vec.astype(np.float32)


def scenario_db_theft(lab: Lab, defender: Defender, rng: np.random.Generator) -> Dict[str, Dict]:
    view = db_only_view(lab, "db_theft")
    conn = db.connect(view.db_path)
    dim, out = lab.cfg.embedding_dim, {}
    for scheme in SCHEMES:
        blobs = {u: b for u, _v, b in db.iter_active_templates(conn, scheme)}
        stolen = [naive_parse(blobs[u], dim) for u in lab.victims]
        usable = np.array([v is not None for v in stolen])
        forged = np.stack([v if v is not None else lab.attacker_probes[i % len(lab.attacker_probes)]
                           for i, v in enumerate(stolen)])
        out[scheme] = {
            "recoverable_fraction": float(usable.mean()),
            "asr_stolen_data": _mixed_asr(defender, scheme, forged, usable, rng),
            "baseline_far": defender.zero_effort_asr(scheme, rng)}
    conn.close()
    return out


def _mixed_asr(defender: Defender, scheme: str, forged: np.ndarray, usable: np.ndarray,
               rng: np.random.Generator) -> float:
    """ASR when some victims have usable stolen data and the rest fall back to zero-effort."""
    hits = 0.0
    if usable.any():
        rows = np.where(usable)[0]
        hits += defender.asr(scheme, forged, rows) * len(rows)
    if (~usable).any():
        rows = np.where(~usable)[0]
        hits += defender.zero_effort_asr(scheme, rng, rows) * len(rows)
    return float(hits / len(usable))


def _open_with_master(lab: Lab, scheme: str, view):
    """Matcher an attacker holding DB + master key can build (all templates decrypted)."""
    conn = db.connect(view.db_path)
    matcher = open_gallery(conn, scheme, view.keystore(), lab.cfg.embedding_dim,
                           lab.cfg.cancelable_bits)
    conn.close()
    return matcher


def _forged_embeddings(lab: Lab, matcher) -> np.ndarray:
    """One forged embedding per victim from decrypted material: the vector itself (float
    schemes) or the best linear reconstruction from the code (cancelable)."""
    rows = [matcher.index[u] for u in lab.victims]
    if matcher.scheme != "cancelable":
        return matcher.gallery[rows].astype(np.float32)
    return np.stack([cancelable.reconstruct(matcher.gallery[r], matcher.signs[r], matcher.proj)
                     for r in rows])


def _credentials_with_master(lab: Lab, scheme: str, view) -> np.ndarray:
    return _forged_embeddings(lab, _open_with_master(lab, scheme, view))


def _single_user_open(scheme: str, blob: bytes, target: str, version: int,
                      keys: UserKeyMaterial, dim: int) -> bool:
    """Can ONE user's key material open `target`'s template? (True = exposed.)"""
    aad = aad_for(target, scheme, version)
    try:
        if scheme == "plaintext":
            return len(blob) == 4 * dim
        if scheme == "aes":
            return len(aes_gcm_decrypt(keys.aes_key, blob, aad)) == 4 * dim
        if scheme == "ecies":
            return len(ecies_decrypt(keys.ecies_private, blob, aad)) == 4 * dim
        return len(aes_gcm_decrypt(keys.cancel_wrap_key, blob, aad)) > 0
    except TemplateIntegrityError:
        return False


def scenario_key_compromise(lab: Lab, defender: Defender, rng: np.random.Generator,
                            n_single: int = 25) -> Dict[str, Dict]:
    dim = lab.cfg.embedding_dim
    view = master_key_view(lab)                      
    out: Dict[str, Dict] = {}
    plain_thr = lab.thresholds["plaintext"]
    sample = rng.choice(len(lab.victims), size=min(n_single, len(lab.victims)), replace=False)
    conn = db.connect(view.db_path)
    ks_att = view.keystore()
    for scheme in SCHEMES:
        forged = _credentials_with_master(lab, scheme, view)
        rec_cos = _cos(forged, lab.templates)
        # blast radius with the MASTER key: how many templates can actually be decrypted
        blast_master = float(np.mean([
            _opens_with_master(scheme, conn, ks_att, u, lab) for u in lab.victims]))
        # single user's key material
        blobs = {u: (v, b) for u, v, b in db.iter_active_templates(conn, scheme)}
        exposed, asr_hits = [], []
        for pos in sample:
            victim = lab.victims[pos]
            keys = leak_user_keys(lab.ks, victim, blobs[victim][0], dim)
            opened = [_single_user_open(scheme, blobs[t][1], t, blobs[t][0], keys, dim)
                      for t in lab.victims]
            exposed.append(float(np.mean(opened)))
            own = _single_user_open(scheme, blobs[victim][1], victim, blobs[victim][0], keys, dim)
            if scheme == "cancelable":      
                asr_hits.append(defender.zero_effort_asr(scheme, rng, np.array([pos]))) 
            else:
                asr_hits.append(float(own))
        out[scheme] = {
            "asr_master_key": defender.asr(scheme, forged),
            "reconstruction_cosine_master_key": float(rec_cos.mean()),
            "cross_system_asr_master_key": float((rec_cos >= plain_thr).mean()),
            "asr_single_user_key": float(np.mean(asr_hits)),
            "blast_radius_single_user_key": float(np.mean(exposed)),
            "blast_radius_master_key": blast_master}
    conn.close()
    return out


def _opens_with_master(scheme: str, conn, ks, user_id: str, lab: Lab) -> float:
    try:
        open_user(conn, scheme, ks, user_id, lab.cfg.embedding_dim, lab.cfg.cancelable_bits)
        return 1.0
    except (TemplateIntegrityError, KeyError):
        return 0.0

def scenario_tampering(lab: Lab, defender: Defender, rng: np.random.Generator) -> Dict[str, Dict]:
    dim, bits = lab.cfg.embedding_dim, lab.cfg.cancelable_bits
    out: Dict[str, Dict] = {}
    for scheme in SCHEMES:
        # (a) flip one random byte of every victim's blob
        flip = db.connect(lab.copy_db("tamper_flip"))
        detected = 0
        for u in lab.victims:
            version, blob = db.get_active_template(flip, u, scheme)
            bad = bytearray(blob)
            bad[int(rng.integers(0, len(bad)))] ^= 0xFF
            flip.execute("UPDATE templates SET blob = ? WHERE user_id = ? AND scheme = ? "
                         "AND key_version = ?", (bytes(bad), u, scheme, version))
            flip.commit()
            try:
                open_user(flip, scheme, lab.ks, u, dim, bits)
            except TemplateIntegrityError:
                detected += 1
        flip.close()
        # (b) overwrite the victim's row with a blob the attacker legitimately created for
        #     their OWN account, then log in as the victim with their own face
        sub = db.connect(lab.copy_db("tamper_sub"))
        sub_detected = sub_success = 0
        for i, u in enumerate(lab.victims):
            face = lab.attacker_probes[i % len(lab.attacker_probes)]
            version, _ = db.get_active_template(sub, u, scheme)
            own_blob = protect(scheme, lab.ks, f"attacker_{i}", 1, face, bits)
            sub.execute("UPDATE templates SET blob = ? WHERE user_id = ? AND scheme = ? "
                        "AND key_version = ?", (own_blob, u, scheme, version))
            sub.commit()
            try:
                one = open_user(sub, scheme, lab.ks, u, dim, bits)
            except TemplateIntegrityError:
                sub_detected += 1
                continue
            sub_success += int(verify(one, face, u, lab.thresholds[scheme]).accepted)
        sub.close()
        n = len(lab.victims)
        out[scheme] = {"detection_rate_bitflip": detected / n,
                       "detection_rate_substitution": sub_detected / n,
                       "asr_substitution": sub_success / n}
    return out


def scenario_replay(lab: Lab, defender: Defender, rng: np.random.Generator) -> Dict[str, Dict]:
    dim, bits = lab.cfg.embedding_dim, lab.cfg.cancelable_bits
    out: Dict[str, Dict] = {}
    for scheme in SCHEMES:
        conn = db.connect(lab.copy_db("replay"))
        unprotected = protected = genuine = stale = 0
        for i, u in enumerate(lab.victims):
            probe, thr = lab.genuine_probes[i], lab.thresholds[scheme]
            one = open_user(conn, scheme, lab.ks, u, dim, bits)
            nonce, issued = uuid.uuid4().hex, time.time()
            first_ok, _ = check_request_freshness(conn, nonce, issued)
            genuine_ok = first_ok and verify(one, probe, u, thr).accepted
            genuine += int(genuine_ok)
            if not genuine_ok:
                continue                                    
            unprotected += int(verify(one, probe, u, thr).accepted)     
            replay_ok, _ = check_request_freshness(conn, nonce, issued)  
            protected += int(replay_ok and verify(one, probe, u, thr).accepted)
            old_ok, _ = check_request_freshness(conn, uuid.uuid4().hex, time.time() - 3600)
            stale += int(not old_ok)
        conn.close()
        n_ok = max(genuine, 1)
        out[scheme] = {"replay_accept_rate_unprotected": unprotected / n_ok,
                       "replay_accept_rate_protected": protected / n_ok,
                       "stale_reject_rate": stale / n_ok,
                       "genuine_accept_rate": genuine / len(lab.victims)}
    return out

def _code_scores(codes_a: np.ndarray, codes_b: np.ndarray) -> np.ndarray:
    """Template-domain score between two aligned stacks of cancelable codes."""
    agree = (codes_a.astype(np.float32) * codes_b.astype(np.float32)).mean(axis=1)
    return cancelable.score_from_agreement(agree)


def scenario_revocation(lab: Lab, defender: Defender, rng: np.random.Generator) -> Dict[str, Dict]:
    """Credential = what the attacker extracted with the master key BEFORE the re-issue:
    float schemes -> the decrypted vector; cancelable -> the stolen binary code, injected in the
    template domain (the classical stolen-template attack). The reconstructed embedding is
    reported separately because it lives in the biometric domain and is key-independent."""
    bits = lab.cfg.cancelable_bits
    view = master_key_view(lab, "revocation_attacker")
    conn = db.connect(lab.copy_db("revocation"))
    for u, fresh in zip(lab.victims, lab.alt_templates):     
        reissue_user(conn, lab.ks, u, fresh, bits)
    after = {s: open_gallery(conn, s, lab.ks, lab.cfg.embedding_dim, bits) for s in SCHEMES}
    conn.close()
    out: Dict[str, Dict] = {}
    for scheme in SCHEMES:
        stolen = _open_with_master(lab, scheme, view)
        rows = [stolen.index[u] for u in lab.victims]
        idx_new = np.array([after[scheme].index[u] for u in lab.victims])
        thr = lab.thresholds[scheme]
        genuine = score_pairs(after[scheme], lab.genuine_probes, idx_new)
        embeddings = _forged_embeddings(lab, stolen)
        recon_after = score_pairs(after[scheme], embeddings, idx_new)
        if scheme == "cancelable":
            old_codes = stolen.gallery[rows]
            before = _code_scores(old_codes, defender.matchers[scheme].gallery[defender.idx])
            after_s = _code_scores(old_codes, after[scheme].gallery[idx_new])
        else:
            before = score_pairs(defender.matchers[scheme], embeddings, defender.idx)
            after_s = recon_after
        out[scheme] = {"asr_old_credential_before_reissue": float((before >= thr).mean()),
                       "asr_old_credential_after_reissue": float((after_s >= thr).mean()),
                       "genuine_accept_after_reissue": float((genuine >= thr).mean()),
                       "asr_reconstructed_embedding_after_reissue": float((recon_after >= thr).mean())}
    return out


def _bytes_similarity(a: bytes, b: bytes) -> float:
    """Best generic attempt to link two opaque blobs: correlation of their byte values."""
    n = min(len(a), len(b))
    x = np.frombuffer(a[:n], dtype=np.uint8).astype(np.float64)
    y = np.frombuffer(b[:n], dtype=np.uint8).astype(np.float64)
    x, y = x - x.mean(), y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom else 0.0


def scenario_unlinkability(lab: Lab, defender: Defender, rng: np.random.Generator,
                           non_mated_per_victim: int = 10) -> Dict[str, Dict]:
    dim, bits = lab.cfg.embedding_dim, lab.cfg.cancelable_bits
    conn_b = db.connect(lab.workdir / "system_b.sqlite")          
    db.init_schema(conn_b)
    enroll_users(conn_b, lab.ks, lab.victims, lab.alt_templates, bits, version=2)
    conn_a = lab.connect()
    n = len(lab.victims)
    out: Dict[str, Dict] = {}
    for scheme in SCHEMES:
        blobs_a = {u: b for u, _v, b in db.iter_active_templates(conn_a, scheme)}
        blobs_b = {u: b for u, _v, b in db.iter_active_templates(conn_b, scheme)}
        open_a = open_gallery(conn_a, scheme, lab.ks, dim, bits)
        open_b = open_gallery(conn_b, scheme, lab.ks, dim, bits)

        def link_scores(sim) -> tuple:
            mated = np.array([sim(i, i) for i in range(n)])
            others = rng.integers(0, n - 1, size=(n, non_mated_per_victim))
            others = others + (others >= np.arange(n)[:, None])           
            non_mated = np.array([[sim(i, j) for j in row] for i, row in enumerate(others)]).ravel()
            return mated, non_mated

        def auc(mated: np.ndarray, non_mated: np.ndarray) -> float:
            """Strongest simple linker: the raw score, or its magnitude (templates of one person
            under different keys are not exactly independent, so |score| can leak a little)."""
            y = np.r_[np.ones(len(mated)), np.zeros(len(non_mated))]
            both = np.r_[mated, non_mated]
            return float(max(roc_auc_score(y, both), roc_auc_score(y, np.abs(both))))

        users = lab.victims
        stored = link_scores(lambda i, j: _bytes_similarity(blobs_a[users[i]], blobs_b[users[j]])
                             if scheme != "plaintext" else
                             float(np.frombuffer(blobs_a[users[i]], "<f4") @
                                   np.frombuffer(blobs_b[users[j]], "<f4")))
        if scheme == "cancelable":
            dec = link_scores(lambda i, j: cancelable.code_score(
                open_a.gallery[open_a.index[users[i]]], open_b.gallery[open_b.index[users[j]]]))
        else:
            dec = link_scores(lambda i, j: float(open_a.gallery[open_a.index[users[i]]] @
                                                 open_b.gallery[open_b.index[users[j]]]))
        out[scheme] = {"linkage_auc_stored": auc(*stored), "linkage_auc_decrypted": auc(*dec),
                       "mean_sim_mated_decrypted": float(dec[0].mean()),
                       "mean_sim_nonmated_decrypted": float(dec[1].mean())}
    conn_a.close()
    conn_b.close()
    return out


SCENARIO_RUNNERS: Dict[str, Callable] = {
    "db_theft": scenario_db_theft, "key_compromise": scenario_key_compromise,
    "tampering": scenario_tampering, "replay": scenario_replay,
    "revocation": scenario_revocation, "unlinkability": scenario_unlinkability,
}


def run_all(lab: Lab, scenarios: Optional[Sequence[str]] = None, seed: int = 0) -> Dict:
    unknown = set(scenarios or []) - set(SCENARIO_RUNNERS)
    if unknown:
        raise ValueError(f"unknown scenarios: {sorted(unknown)}")
    defender = Defender(lab)
    results: Dict[str, Dict] = {}
    for sid in scenarios or list(SCENARIO_RUNNERS):
        rng = np.random.default_rng([seed, zlib.crc32(sid.encode())])
        t0 = time.time()
        per_scheme = SCENARIO_RUNNERS[sid](lab, defender, rng)
        meta = SCENARIOS[sid]
        results[sid] = {"title": meta.title, "attacker": meta.attacker, "goal": meta.goal,
                        "note": meta.note, "metrics": list(meta.metrics),
                        "per_scheme": per_scheme, "seconds": round(time.time() - t0, 2)}
    return {"meta": {"n_victims": len(lab.victims), "seed": seed,
                     "far_target": lab.far_target, "thresholds": lab.thresholds,
                     "lab_key_fingerprint": lab.ks.fingerprint()},
            "scenarios": results}


def print_results(res: Dict) -> None:
    for sid, sc in res["scenarios"].items():
        print(f"\n== {sc['title']} ({sid})")
        keys = list(next(iter(sc["per_scheme"].values())))
        print(f"{'metric':<38}" + "".join(f"{s:>12}" for s in SCHEMES))
        for k in keys:
            print(f"{k:<38}" + "".join(f"{sc['per_scheme'][s][k]:>12.3f}" for s in SCHEMES))


def main(argv: Optional[list] = None) -> int:
    from src.Threat_Scenario_Setup import build_lab
    p = argparse.ArgumentParser(description="Run the security scenarios against all schemes")
    p.add_argument("--victims", type=int, default=200)
    p.add_argument("--scenarios", nargs="+", choices=list(SCENARIO_RUNNERS), default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    paths = get_paths().ensure()
    lab = build_lab(paths, n_victims=args.victims, seed=args.seed)
    res = run_all(lab, args.scenarios, args.seed)
    write_json(paths.file("security_results.json"), res)
    print_results(res)
    print(f"\nWrote {paths.file('security_results.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
