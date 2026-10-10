from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from common import db
from common.artifacts import load_embeddings, write_json
from common.config import SCHEMES, PipelineConfig, Paths, get_paths, load_pipeline_config
from common.crypto import KeyStore
from src.Embedding_Encryption import build_user_templates, enroll_users
from src.Query_Authorisation import open_gallery, score_pairs


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    attacker: str          
    goal: str
    metrics: Tuple[str, ...]
    note: str = ""


SCENARIOS: Dict[str, Scenario] = {s.id: s for s in [
    Scenario(
        "db_theft", "Database theft",
        "Copies the SQLite file. Has no key.",
        "Recover biometric data, or use the stolen data to log in as a victim.",
        ("recoverable_fraction", "asr_stolen_data", "baseline_far"),
        "ASR = attack success rate: share of victims for whom the forged credential is accepted. "
        "baseline_far is the zero-effort rate of an attacker using their own face."),
    Scenario(
        "key_compromise", "Key compromise",
        "Has the database AND either the master key (everything) or one user's derived keys.",
        "Decrypt templates and impersonate victims; measure how far the damage spreads.",
        ("asr_master_key", "reconstruction_cosine_master_key", "cross_system_asr_master_key",
         "asr_single_user_key", "blast_radius_single_user_key", "blast_radius_master_key"),
        "reconstruction_cosine is how well the attacker rebuilds the victim's embedding. "
        "cross_system_asr is the rebuilt embedding used against an unprotected system. "
        "For cancelable templates a single user's key does not include the shared projection."),
    Scenario(
        "tampering", "Template tampering",
        "Can write to the database but has no key.",
        "Corrupt a victim's template, or overwrite it with a template of their own face.",
        ("detection_rate_bitflip", "detection_rate_substitution", "asr_substitution"),
        "Substitution copies the attacker's own enrolled blob into the victim's row."),
    Scenario(
        "replay", "Replay of a captured request",
        "Recorded a genuine authentication request (query + nonce + timestamp) in transit.",
        "Be accepted by re-sending the identical request.",
        ("replay_accept_rate_unprotected", "replay_accept_rate_protected", "stale_reject_rate"),
        "Replay protection is a protocol feature (nonce + time window), so it does not depend on "
        "the template scheme; the same numbers for every scheme are expected."),
    Scenario(
        "revocation", "Revocation and re-issue",
        "Holds a credential extracted earlier; the victim then re-enrols with a new key version.",
        "Keep authenticating with the old credential after the compromise was handled.",
        ("asr_old_credential_before_reissue", "asr_old_credential_after_reissue",
         "genuine_accept_after_reissue", "asr_reconstructed_embedding_after_reissue"),
        "Credential = decrypted vector (AES/ECIES) or stolen binary code (cancelable). Only "
        "cancelable templates can actually be cancelled: re-keying AES/ECIES re-encrypts the same "
        "face, so a decrypted vector keeps working. An embedding RECONSTRUCTED from a stolen code "
        "is a biometric-domain input, so it is reported separately and is not revoked by re-keying."),
    Scenario(
        "unlinkability", "Cross-system linkage",
        "Sees one person's template in two systems (two key versions).",
        "Decide whether two templates belong to the same person.",
        ("linkage_auc_stored", "linkage_auc_decrypted", "mean_sim_mated_decrypted",
         "mean_sim_nonmated_decrypted"),
        "AUC 0.5 = unlinkable, 1.0 = always linkable. 'decrypted' assumes the encryption layer "
        "was removed, which isolates the effect of the cancelable transform."),
]}



@dataclass
class Lab:
    cfg: PipelineConfig
    ks: KeyStore                    
    workdir: Path
    db_path: Path
    victims: List[str]
    templates: np.ndarray           
    genuine_probes: np.ndarray      
    alt_templates: np.ndarray       
    attacker_probes: np.ndarray     
    thresholds: Dict[str, float] = field(default_factory=dict)   
    far_target: float = 0.01

    def connect(self):
        return db.connect(self.db_path)

    def copy_db(self, name: str) -> Path:
        """A private copy of the lab database for a destructive experiment."""
        dest = self.workdir / f"{name}.sqlite"
        shutil.copy2(self.db_path, dest)
        return dest


def _mean_unit(rows: np.ndarray) -> np.ndarray:
    v = rows.mean(axis=0)
    return (v / np.linalg.norm(v)).astype(np.float32)


def calibrate_thresholds(lab: Lab, n_pairs: int = 20000, seed: int = 0) -> Dict[str, float]:
    """Per-scheme threshold such that zero-effort impostors are accepted at `far_target`."""
    rng = np.random.default_rng(seed)
    n = min(n_pairs, len(lab.attacker_probes) * len(lab.victims))
    att = rng.integers(0, len(lab.attacker_probes), n)
    vic = rng.integers(0, len(lab.victims), n)
    conn = lab.connect()
    out = {}
    for scheme in SCHEMES:
        matcher = open_gallery(conn, scheme, lab.ks, lab.cfg.embedding_dim, lab.cfg.cancelable_bits)
        idx = np.array([matcher.index[lab.victims[v]] for v in vic])
        scores = score_pairs(matcher, lab.attacker_probes[att], idx)
        out[scheme] = float(np.quantile(scores, 1.0 - lab.far_target))
    conn.close()
    return out


def build_lab(paths: Optional[Paths] = None, cfg: Optional[PipelineConfig] = None,
              n_victims: int = 200, seed: int = 0, workdir: Optional[Path] = None,
              max_attackers: int = 2000, far_target: float = 0.01) -> Lab:
    paths = paths or get_paths()
    cfg = cfg or load_pipeline_config(paths)
    emb, splits = load_embeddings(paths)
    all_ids, all_vecs = build_user_templates(emb, splits)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(all_ids), size=min(n_victims, len(all_ids)), replace=False))
    victims = [all_ids[i] for i in pick]
    templates = all_vecs[pick]

    known = splits[splits.group == "known"].copy()
    known["uid"] = known["identity"].map(lambda i: f"id_{int(i):05d}")
    genuine, alt = [], []
    for uid in victims:
        mine = known[known.uid == uid]
        test = mine[mine.partition == "test"]["row"].to_numpy()
        val = mine[mine.partition == "val"]["row"].to_numpy()
        pool_g = test if len(test) else val
        pool_a = val if len(val) else test
        genuine.append(emb[rng.choice(pool_g)])
        alt.append(_mean_unit(emb[rng.permutation(pool_a)[:3]]))
    unknown_rows = splits[(splits.group == "unknown") & (splits.partition == "test")]["row"].to_numpy()
    attackers = emb[rng.permutation(unknown_rows)[:max_attackers]]

    workdir = Path(workdir) if workdir else paths.artifacts / "lab"
    workdir.mkdir(parents=True, exist_ok=True)
    master = hashlib.sha256(f"capstone-lab|{seed}".encode()).digest()   # lab-only key
    ks = KeyStore(master)
    db_path = workdir / "lab.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = db.connect(db_path)
    db.init_schema(conn)
    enroll_users(conn, ks, victims, templates, cfg.cancelable_bits)
    conn.close()

    lab = Lab(cfg=cfg, ks=ks, workdir=workdir, db_path=db_path, victims=victims,
              templates=templates, genuine_probes=np.stack(genuine),
              alt_templates=np.stack(alt), attacker_probes=attackers, far_target=far_target)
    lab.thresholds = calibrate_thresholds(lab, seed=seed)
    return lab


@dataclass(frozen=True)
class UserKeyMaterial:
    """Everything derived for ONE user. Note: no shared cancelable projection."""
    user_id: str
    version: int
    aes_key: bytes
    ecies_private: object              
    cancel_wrap_key: bytes
    cancel_signs: np.ndarray


@dataclass
class AttackerView:
    name: str
    db_path: Path                                   
    master_key: Optional[bytes] = None              
    user_keys: Dict[str, UserKeyMaterial] = field(default_factory=dict)

    def keystore(self) -> Optional[KeyStore]:
        return KeyStore(self.master_key) if self.master_key else None


def leak_user_keys(ks: KeyStore, user_id: str, version: int, dim: int) -> UserKeyMaterial:
    return UserKeyMaterial(user_id, version, ks.aes_key(user_id, version),
                           ks.ecies_private(user_id, version),
                           ks.cancelable_wrap_key(user_id, version),
                           ks.cancelable_signs(user_id, version, dim))


def db_only_view(lab: Lab, name: str = "db_theft") -> AttackerView:
    return AttackerView(name, lab.copy_db(f"attacker_{name}"))


def master_key_view(lab: Lab, name: str = "master_key") -> AttackerView:
    return AttackerView(name, lab.copy_db(f"attacker_{name}"), master_key=lab.ks._master)


def single_user_view(lab: Lab, user_id: str, name: Optional[str] = None) -> AttackerView:
    name = name or f"user_key_{user_id}"
    conn = lab.connect()
    version = db.get_user(conn, user_id)["key_version"]
    conn.close()
    return AttackerView(name, lab.copy_db(f"attacker_{name}"),
                        user_keys={user_id: leak_user_keys(lab.ks, user_id, version,
                                                           lab.cfg.embedding_dim)})


def export_threat_model(paths: Paths) -> Path:
    out = paths.file("threat_model.json")
    write_json(out, [asdict(s) for s in SCENARIOS.values()])
    return out


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Show threat scenarios and build a small test lab")
    p.add_argument("--victims", type=int, default=50)
    args = p.parse_args(argv)
    paths = get_paths().ensure()
    print(f"Wrote {export_threat_model(paths)}")
    for s in SCENARIOS.values():
        print(f"\n[{s.id}] {s.title}\n  attacker: {s.attacker}\n  goal    : {s.goal}\n"
              f"  metrics : {', '.join(s.metrics)}")
    lab = build_lab(paths, n_victims=args.victims)
    print(f"\nLab: {len(lab.victims)} victims, {len(lab.attacker_probes)} attacker probes, "
          f"thresholds (FAR {lab.far_target:.0%}): "
          + ", ".join(f"{k}={v:.3f}" for k, v in lab.thresholds.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
