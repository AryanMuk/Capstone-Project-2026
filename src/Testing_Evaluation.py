"""
Testing & Evaluation — owner: Ayati   (Gantt: Testing & Evaluation, 6 Oct)
Branch: feature/testing-evaluation-ayati
Depends on: Embedding_Encryption_Suvrat (templates.sqlite), Query_Authorisation_Jaskaran,
            splits.csv, embeddings.npy
Produces  : eval_results.json, figures/*.png      (also tests/test_smoke_Ayati.py)

Protocols, all computed from ONE score matrix per scheme (probes x enrolled users):
  closed-set  : known test probes; Rank-1, Rank-5, CMC.                (ties count against us)
  verification: genuine = probe vs own template; impostor = known probes vs other users plus
                unknown probes vs random users; FAR, FRR, EER, AUC-ROC.
  open-set    : known + unknown test probes, top-1 decision; DIR at fixed false-positive
                identification rates (FPIR).
Thresholds are chosen on the VAL partition and applied unchanged to TEST.

Run (repo root):
    python -m src.Testing_Evaluation_Ayati evaluate [--max-probes 20000] [--schemes aes cancelable]
    python -m src.Testing_Evaluation_Ayati smoke      # run the milestone smoke tests

Sections / commits (use `git add -p` to commit section by section):
  1. feat(eval): add CMC, ROC/EER/AUC and open-set DIR metrics with val-chosen thresholds
  2. feat(eval): evaluate each scheme (accuracy, latency, storage) from one score matrix
  3. feat(eval): add ROC/CMC/efficiency figures
  4. feat(eval): add evaluate/smoke command-line entry point
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from common import db
from common.artifacts import load_embeddings, read_json, user_id_for, write_json
from common.config import REPO_ROOT, SCHEMES, PipelineConfig, Paths, get_paths, load_pipeline_config
from common.crypto import KeyStore
from src.Query_Authorisation_Jaskaran import (GalleryMatcher, identify, open_gallery, open_user,
                                              score_matrix, verify)

OPEN_SET_FPIRS = (0.001, 0.01, 0.1)

# =============================================================================
# SECTION 1/4 — Metrics
# COMMIT: feat(eval): add CMC, ROC/EER/AUC and open-set DIR metrics with val-chosen thresholds
# =============================================================================


def ranks_of_truth(scores: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    """1-based rank of the true user per probe. Ties are counted AGAINST the system
    (rank = number of users scoring >= the true user), so the result is never optimistic."""
    true_score = scores[np.arange(len(scores)), true_idx]
    return (scores >= true_score[:, None]).sum(axis=1)


def cmc_curve(ranks: np.ndarray, max_rank: int = 20) -> np.ndarray:
    return np.array([(ranks <= r).mean() for r in range(1, max_rank + 1)])


def sample_impostor_scores(scores: np.ndarray, true_idx: Optional[np.ndarray], per_probe: int,
                           rng: np.random.Generator) -> np.ndarray:
    """Pick `per_probe` wrong-user columns per probe. true_idx=None means every user is wrong
    (unknown identities)."""
    m, u = scores.shape
    if u < 2:
        raise ValueError("need at least 2 enrolled users")
    cols = rng.integers(0, u, size=(m, per_probe))
    if true_idx is not None:
        clash = cols == true_idx[:, None]
        cols[clash] = (cols[clash] + 1) % u
    return np.take_along_axis(scores, cols, axis=1).ravel()


def far_frr(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> Tuple[float, float]:
    """Accept when score >= threshold."""
    return float((impostor >= threshold).mean()), float((genuine < threshold).mean())


def threshold_at_far(impostor: np.ndarray, far: float) -> float:
    return float(np.quantile(impostor, 1.0 - far))


def verification_metrics(genuine: np.ndarray, impostor: np.ndarray) -> Dict:
    y = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    s = np.concatenate([genuine, impostor])
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return {"auc": float(roc_auc_score(y, s)), "eer": float((fpr[i] + fnr[i]) / 2),
            "eer_threshold": float(thr[i]), "fpr": fpr, "tpr": tpr}


def downsample_roc(fpr: np.ndarray, tpr: np.ndarray, points: int = 200) -> Dict[str, List[float]]:
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(points, len(fpr))).astype(int))
    return {"far": fpr[keep].round(6).tolist(), "tpr": tpr[keep].round(6).tolist()}


def open_set_report(val_known_top: np.ndarray, test_known_top: np.ndarray,
                    test_known_correct: np.ndarray, val_unknown_top: np.ndarray,
                    test_unknown_top: np.ndarray) -> Dict[str, Dict[str, float]]:
    """DIR (correct and accepted) vs FPIR (unknown accepted). Threshold per FPIR target is
    taken from VAL unknown probes; DIR/FPIR are then measured on TEST."""
    out = {}
    for target in OPEN_SET_FPIRS:
        t = threshold_at_far(val_unknown_top, target)
        out[str(target)] = {
            "threshold": t,
            "fpir_test": float((test_unknown_top >= t).mean()),
            "dir_test": float(((test_known_top >= t) & test_known_correct).mean()),
        }
    return out


# =============================================================================
# SECTION 2/4 — Scheme evaluation
# COMMIT: feat(eval): evaluate each scheme (accuracy, latency, storage) from one score matrix
# =============================================================================


def select_probes(splits: pd.DataFrame, max_probes: int, seed: int) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Same probes for every scheme: {(group, partition): rows with identity,row}."""
    rng = np.random.default_rng(seed)
    out = {}
    for group in ("known", "unknown"):
        for part in ("val", "test"):
            sub = splits[(splits.group == group) & (splits.partition == part)]
            if len(sub) > max_probes:
                sub = sub.iloc[np.sort(rng.choice(len(sub), size=max_probes, replace=False))]
            out[(group, part)] = sub[["identity", "row"]].reset_index(drop=True)
    return out


def _true_index(matcher: GalleryMatcher, identities: np.ndarray) -> np.ndarray:
    return np.array([matcher.index.get(user_id_for(i), -1) for i in identities])


def measure_latency(conn, ks: KeyStore, matcher: GalleryMatcher, emb: np.ndarray,
                    probes: pd.DataFrame, cfg: PipelineConfig, threshold: float,
                    n: int = 50) -> Dict[str, float]:
    """Per-request wall-clock in ms. verify = decrypt ONE template + match;
    identify_match = match against the already-decrypted gallery."""
    scheme = matcher.scheme
    verify_ms, ident_ms = [], []
    for _, row in probes.head(n).iterrows():
        uid, x = user_id_for(row["identity"]), emb[row["row"]]
        t0 = time.perf_counter()
        one = open_user(conn, scheme, ks, uid, cfg.embedding_dim, cfg.cancelable_bits)
        verify(one, x, uid, threshold)
        verify_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        identify(matcher, x, threshold)
        ident_ms.append((time.perf_counter() - t0) * 1000)
    return {"verify_ms_median": float(np.median(verify_ms)),
            "identify_match_ms_median": float(np.median(ident_ms)), "requests_timed": len(verify_ms)}


def evaluate_scheme(scheme: str, conn, ks: KeyStore, emb: np.ndarray, probes: Dict,
                    cfg: PipelineConfig, seed: int, device: Optional[str] = None) -> Dict:
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    matcher = open_gallery(conn, scheme, ks, cfg.embedding_dim, cfg.cancelable_bits)
    open_s = time.perf_counter() - t0

    def matrix(key):
        return score_matrix(matcher, emb[probes[key]["row"].to_numpy()], device=device)

    kv, kt = matrix(("known", "val")), matrix(("known", "test"))
    uv, ut = matrix(("unknown", "val")), matrix(("unknown", "test"))
    true_v = _true_index(matcher, probes[("known", "val")]["identity"].to_numpy())
    true_t = _true_index(matcher, probes[("known", "test")]["identity"].to_numpy())
    ok_v, ok_t = true_v >= 0, true_t >= 0
    kv, true_v, kt, true_t = kv[ok_v], true_v[ok_v], kt[ok_t], true_t[ok_t]
    per = cfg.imp_per_probe

    # closed-set (test)
    ranks = ranks_of_truth(kt, true_t)
    cmc = cmc_curve(ranks)

    # verification: threshold from val, applied to test
    gen_v = kv[np.arange(len(kv)), true_v]
    imp_v = np.concatenate([sample_impostor_scores(kv, true_v, per, rng),
                            sample_impostor_scores(uv, None, per, rng)])
    gen_t = kt[np.arange(len(kt)), true_t]
    imp_t = np.concatenate([sample_impostor_scores(kt, true_t, per, rng),
                            sample_impostor_scores(ut, None, per, rng)])
    val_v = verification_metrics(gen_v, imp_v)
    test_v = verification_metrics(gen_t, imp_t)
    thr_far1 = threshold_at_far(imp_v, 0.01)
    far_e, frr_e = far_frr(gen_t, imp_t, val_v["eer_threshold"])
    far_1, frr_1 = far_frr(gen_t, imp_t, thr_far1)

    # open-set
    open_set = open_set_report(
        kv.max(axis=1), kt.max(axis=1), kt.argmax(axis=1) == true_t,
        uv.max(axis=1), ut.max(axis=1))

    latency = measure_latency(conn, ks, matcher, emb, probes[("known", "test")], cfg, thr_far1)
    stats = db.template_stats(conn).get(scheme, {})
    return {
        "n_users": len(matcher), "rejected_templates": len(matcher.rejected),
        "probes": {"known_test": int(len(kt)), "unknown_test": int(len(ut)),
                   "genuine_pairs_test": int(len(gen_t)), "impostor_pairs_test": int(len(imp_t))},
        "closed_set": {"rank1": float((ranks <= 1).mean()), "rank5": float((ranks <= 5).mean()),
                       "cmc": cmc.round(6).tolist()},
        "verification": {
            "auc": test_v["auc"], "eer": test_v["eer"], "eer_val": val_v["eer"],
            "threshold_eer_from_val": val_v["eer_threshold"],
            "far_at_eer_threshold": far_e, "frr_at_eer_threshold": frr_e,
            "threshold_far1pct_from_val": thr_far1,
            "far_at_far1pct_threshold": far_1, "frr_at_far1pct_threshold": frr_1,
            "roc": downsample_roc(test_v["fpr"], test_v["tpr"])},
        "open_set": open_set,
        "efficiency": {**latency, "gallery_decrypt_s": open_s,
                       "decrypt_ms_per_template": open_s * 1000 / max(len(matcher), 1),
                       "bytes_per_template": stats.get("mean_bytes"),
                       "total_bytes": stats.get("total_bytes")},
    }


def run_evaluation(cfg: PipelineConfig, paths: Optional[Paths] = None,
                   schemes: Sequence[str] = SCHEMES, max_probes: int = 20000, seed: int = 0,
                   device: Optional[str] = None, figures: bool = True) -> Dict:
    from src.Embedding_Encryption_Suvrat import TEMPLATES_DB
    paths = paths or get_paths()
    emb, splits = load_embeddings(paths)
    probes = select_probes(splits, max_probes, seed)
    conn = db.connect(paths.file(TEMPLATES_DB))
    ks = KeyStore.load_or_create(paths, create=False)
    results = {}
    for scheme in schemes:
        print(f"Evaluating {scheme} ...")
        results[scheme] = evaluate_scheme(scheme, conn, ks, emb, probes, cfg, seed, device)
    conn.close()
    timings_file = paths.file("enroll_timings.json")
    if timings_file.exists():
        timings = read_json(timings_file)
        for s in results:
            if s in timings:
                results[s]["efficiency"]["enroll_ms_mean"] = timings[s]["enroll_ms_mean"]
    out = {"meta": {"seed": seed, "max_probes_per_group": max_probes,
                    "n_embeddings": int(len(emb)), "config": cfg.__dict__,
                    "key_fingerprint": ks.fingerprint(),
                    "note": "thresholds chosen on val partition, metrics reported on test"},
           "schemes": results}
    write_json(paths.file("eval_results.json"), out)
    if figures:
        make_figures(out, paths)
    return out


# =============================================================================
# SECTION 3/4 — Figures
# COMMIT: feat(eval): add ROC/CMC/efficiency figures
# =============================================================================
def make_figures(results: Dict, paths: Paths) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = paths.file("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    colours = {"plaintext": "#777777", "aes": "#1f77b4", "ecies": "#ff7f0e", "cancelable": "#2ca02c"}
    written = []

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for s, r in results["schemes"].items():
        roc = r["verification"]["roc"]
        ax.plot(np.maximum(roc["far"], 1e-6), roc["tpr"], label=f"{s} (AUC {r['verification']['auc']:.3f})",
                color=colours[s])
    ax.set_xscale("log")
    ax.set(xlabel="FAR", ylabel="1 - FRR (TPR)", title="Verification ROC (test)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f = fig_dir / "roc.png"
    fig.savefig(f, dpi=150)
    plt.close(fig)
    written.append(str(f))

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for s, r in results["schemes"].items():
        ax.plot(range(1, len(r["closed_set"]["cmc"]) + 1), r["closed_set"]["cmc"], marker="o",
                ms=3, label=s, color=colours[s])
    ax.set(xlabel="Rank", ylabel="Identification rate", title="Closed-set CMC (test)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f = fig_dir / "cmc.png"
    fig.savefig(f, dpi=150)
    plt.close(fig)
    written.append(str(f))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    names = list(results["schemes"])
    axes[0].bar(names, [results["schemes"][s]["efficiency"]["bytes_per_template"] for s in names],
                color=[colours[s] for s in names])
    axes[0].set(title="Bytes per stored template", yscale="log")
    axes[1].bar(names, [results["schemes"][s]["efficiency"]["verify_ms_median"] for s in names],
                color=[colours[s] for s in names])
    axes[1].set(title="Verify latency (ms, median)")
    fig.tight_layout()
    f = fig_dir / "efficiency.png"
    fig.savefig(f, dpi=150)
    plt.close(fig)
    written.append(str(f))
    return written


# =============================================================================
# SECTION 4/4 — Command-line entry point
# COMMIT: feat(eval): add evaluate/smoke command-line entry point
# =============================================================================
def print_summary(results: Dict) -> None:
    print(f"\n{'scheme':<11}{'rank1':>8}{'rank5':>8}{'EER':>8}{'AUC':>8}{'verify ms':>11}{'bytes':>8}")
    for s, r in results["schemes"].items():
        print(f"{s:<11}{r['closed_set']['rank1']:>8.3f}{r['closed_set']['rank5']:>8.3f}"
              f"{r['verification']['eer']:>8.3f}{r['verification']['auc']:>8.3f}"
              f"{r['efficiency']['verify_ms_median']:>11.2f}"
              f"{(r['efficiency']['bytes_per_template'] or 0):>8.0f}")


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate all schemes / run smoke tests")
    p.add_argument("command", choices=["evaluate", "smoke"], nargs="?", default="evaluate")
    p.add_argument("--schemes", nargs="+", choices=SCHEMES, default=list(SCHEMES))
    p.add_argument("--max-probes", type=int, default=20000,
                   help="cap per (known|unknown) x (val|test) group; limits 1:N cancelable cost")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda or cpu for the cancelable 1:N scoring")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    if args.command == "smoke":
        return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests/test_smoke_Ayati.py"],
                               cwd=str(REPO_ROOT))
    results = run_evaluation(load_pipeline_config(), schemes=args.schemes,
                             max_probes=args.max_probes, seed=args.seed, device=args.device,
                             figures=not args.no_figures)
    print_summary(results)
    print(f"\nWrote {get_paths().file('eval_results.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
