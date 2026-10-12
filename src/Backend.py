from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from common import db
from common.artifacts import images_dir, load_splits, read_json, user_id_for, write_json
from common.config import REPO_ROOT, SCHEMES, PipelineConfig, Paths, get_paths, load_pipeline_config
from common.crypto import KeyStore, TemplateIntegrityError
from src.Data_Preprocessing import ImageProcessor, ProcessResult, Stage
from src.Embedding_Encryption import enroll_users, reissue_user
from src.Query_Authorisation import (check_request_freshness, identify, open_gallery,
                                              open_user, score_matrix, score_pairs, verify)

DEMO_DB = "demo.sqlite"
DEMO_THRESHOLDS = "demo_thresholds.json"
DASHBOARD_DIST = REPO_ROOT / "frontend" / "dist"
DEFAULT_THRESHOLD = {"verify": 0.5, "identify": 0.5}   

class FaceNetEmbedder:
    """face (160,160,3) uint8 -> unit 512-d embedding. Loads FaceNet lazily on first use."""

    def __init__(self, device: str = "cpu"):
        self.device_name, self._model, self._device = device, None, None

    def _load(self):
        if self._model is None:
            from src.Embedding_Generation import get_device, load_facenet
            self._device = get_device(self.device_name)
            self._model = load_facenet(self._device)
        return self._model, self._device

    def batch(self, faces: Sequence[np.ndarray]) -> np.ndarray:
        import torch
        from src.Embedding_Generation import embed_uint8
        model, device = self._load()
        out = [embed_uint8(model, torch.from_numpy(np.stack(faces[i:i + 64])), device)
               for i in range(0, len(faces), 64)]
        return np.concatenate(out)

    def __call__(self, face: np.ndarray) -> np.ndarray:
        return self.batch([face])[0]


def _mean_unit(rows: np.ndarray) -> np.ndarray:
    v = rows.mean(axis=0)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _process_files(paths_list: Sequence[Path], processor: ImageProcessor, embedder,
                   label: str) -> tuple:
    """Run files through the upload processor, embed the accepted ones.
    Returns (embeddings (n,512), kept_indices, reject_reasons dict)."""
    from tqdm import tqdm
    faces, kept, rejects = [], [], {}
    for i, p in enumerate(tqdm(paths_list, desc=label)):
        res = processor.process(Path(p).read_bytes())
        if res.ok:
            faces.append(res.face)
            kept.append(i)
        else:
            rejects[res.reason] = rejects.get(res.reason, 0) + 1
    emb = embedder.batch(faces) if faces else np.zeros((0, 512), np.float32)
    return emb, kept, rejects


def calibrate_thresholds(conn: sqlite3.Connection, ks: KeyStore, cfg: PipelineConfig,
                         probe_vecs: np.ndarray, probe_users: Sequence[str],
                         unknown_vecs: np.ndarray, far: float = 0.01,
                         claims_per_unknown: int = 20, seed: int = 0) -> Dict[str, Dict]:
    """Per scheme: `verify` threshold = impostor-claim score at FAR `far`; `identify` threshold
    = top-1 score of unknown people at FPIR `far` (never lower than the verify threshold)."""
    rng = np.random.default_rng(seed)
    out = {}
    for scheme in SCHEMES:
        m = open_gallery(conn, scheme, ks, cfg.embedding_dim, cfg.cancelable_bits)
        genuine = score_pairs(m, probe_vecs, np.array([m.index[u] for u in probe_users]))
        claims = rng.integers(0, len(m), size=len(unknown_vecs) * claims_per_unknown)
        impostor = score_pairs(m, np.repeat(unknown_vecs, claims_per_unknown, axis=0), claims)
        t_verify = float(np.quantile(impostor, 1.0 - far))
        top1 = score_matrix(m, unknown_vecs).max(axis=1)
        t_ident = max(float(np.quantile(top1, 1.0 - far)), t_verify)
        out[scheme] = {"verify": t_verify, "identify": t_ident,
                       "genuine_accept_rate_at_verify": float((genuine >= t_verify).mean()),
                       "n_genuine": int(len(genuine)), "n_impostor": int(len(impostor)),
                       "n_unknown": int(len(unknown_vecs))}
    return out


def build_demo(paths: Paths, cfg: PipelineConfig, images_folder: Path, n_identities: int = 500,
               n_unknown: int = 400, seed: int = 0, processor: Optional[ImageProcessor] = None,
               embedder=None) -> Dict:
    """Enrol `n_identities` known CelebA identities through the SAME processor used for uploads
    (so enrolled and uploaded faces are cropped identically), then calibrate thresholds on
    held-out val probes and on unknown identities."""
    processor = processor or ImageProcessor()
    embedder = embedder or FaceNetEmbedder()
    splits = load_splits(paths)
    rng = np.random.default_rng(seed)
    known = splits[splits.group == "known"]
    ids = np.array(sorted(known.identity.unique()))
    chosen = np.sort(rng.choice(ids, size=min(n_identities, len(ids)), replace=False))

    gal_rows = known[(known.role == "gallery") & known.identity.isin(chosen)]
    emb_g, kept_g, rej_g = _process_files([images_folder / n for n in gal_rows.image], processor,
                                          embedder, "gallery")
    gal_ident = gal_rows.identity.to_numpy()[kept_g]
    users, vectors = [], []
    for ident in chosen:
        rows = emb_g[gal_ident == ident]
        if len(rows) >= 2:                                  
            users.append(user_id_for(ident))
            vectors.append(_mean_unit(rows))
    if not users:
        raise RuntimeError("no identity produced 2+ usable gallery images; check --celeba-dir "
                           f"and processor rejections: {rej_g}")

    val = known[(known.partition == "val") & known.identity.isin([int(u[3:]) for u in users])]
    val = val.groupby("identity").head(1)
    emb_v, kept_v, _ = _process_files([images_folder / n for n in val.image], processor,
                                      embedder, "val probes")
    probe_users = [user_id_for(i) for i in val.identity.to_numpy()[kept_v]]
    unk = splits[(splits.group == "unknown") & (splits.partition == "val")]
    unk = unk.iloc[np.sort(rng.choice(len(unk), size=min(n_unknown, len(unk)), replace=False))]
    emb_u, _, _ = _process_files([images_folder / n for n in unk.image], processor, embedder,
                                 "unknown")

    ks = KeyStore.load_or_create(paths)
    db_path = paths.file(DEMO_DB)
    if db_path.exists():
        db_path.unlink()
    conn = db.connect(db_path)
    db.init_schema(conn)
    enroll_users(conn, ks, users, np.stack(vectors), cfg.cancelable_bits,
                 names={u: f"CelebA #{int(u[3:])}" for u in users})
    thresholds = calibrate_thresholds(conn, ks, cfg, emb_v, probe_users, emb_u, seed=seed)
    conn.close()
    report = {"n_users": len(users), "gallery_rejections": rej_g, "far_target": 0.01,
              "key_fingerprint": ks.fingerprint(), "schemes": thresholds}
    write_json(paths.file(DEMO_THRESHOLDS), report)
    return report


class ServiceError(Exception):
    """Raised for problems the API should report as a 4xx JSON error."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug[:32]


class Services:
    def __init__(self, paths: Optional[Paths] = None, cfg: Optional[PipelineConfig] = None,
                 processor: Optional[ImageProcessor] = None,
                 embedder: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 keystore: Optional[KeyStore] = None, db_path: Optional[Path] = None):
        self.paths = (paths or get_paths()).ensure()
        self.cfg = cfg or load_pipeline_config(self.paths)
        self.processor = processor or ImageProcessor()
        self.embedder = embedder or FaceNetEmbedder()
        self.ks = keystore or KeyStore.load_or_create(self.paths)
        self.db_path = Path(db_path) if db_path else self.paths.file(DEMO_DB)
        conn = self.connect()
        db.init_schema(conn)
        conn.close()
        self._lab = None
        self._lab_lock = threading.Lock()

    # -- plumbing ------------------------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        return db.connect(self.db_path)       # one connection per request: thread-safe

    def thresholds(self) -> Dict[str, Dict]:
        f = self.paths.file(DEMO_THRESHOLDS)
        if f.exists():
            return read_json(f)["schemes"]
        return {s: dict(DEFAULT_THRESHOLD) for s in SCHEMES}

    def calibrated(self) -> bool:
        return self.paths.file(DEMO_THRESHOLDS).exists()

    def names(self, conn: sqlite3.Connection) -> Dict[str, str]:
        return {u["user_id"]: u["display_name"] for u in db.list_users(conn)}

    def _check_scheme(self, scheme: str) -> None:
        if scheme not in SCHEMES:
            raise ServiceError(f"scheme must be one of {list(SCHEMES)}")

    # -- authenticate ----------------------------------------------------------------
    def authenticate(self, data: bytes, scheme: str, mode: str,
                     claimed_user: Optional[str] = None, nonce: Optional[str] = None,
                     issued_at: Optional[float] = None) -> Dict:
        self._check_scheme(scheme)
        if mode not in ("verify", "identify"):
            raise ServiceError("mode must be 'verify' or 'identify'")
        if mode == "verify" and not claimed_user:
            raise ServiceError("verify mode needs claimed_user")
        t_start = time.perf_counter()
        conn = self.connect()
        thr = self.thresholds()[scheme]["verify" if mode == "verify" else "identify"]
        stages: List[Stage] = []
        base = {"scheme": scheme, "mode": mode, "threshold": thr, "score": None, "user_id": None,
                "user_name": None, "top": [], "face_png_b64": None}

        def finish(accepted: bool, reason: str, message: str, **extra) -> Dict:
            latency = (time.perf_counter() - t_start) * 1000
            db.log_auth(conn, scheme, mode, claimed_user, "accepted" if accepted else "rejected",
                        extra.get("score"), reason, latency)
            return {**base, **extra, "accepted": accepted, "reason": reason, "message": message,
                    "stages": [vars(s) for s in stages], "latency_ms": round(latency, 2)}

        try:
            if nonce is not None:                                   # replay protection
                t0 = time.perf_counter()
                fresh, why = check_request_freshness(conn, nonce, issued_at or time.time())
                stages.append(Stage("freshness", fresh, (time.perf_counter() - t0) * 1000, why))
                if not fresh:
                    return finish(False, why, "request rejected by replay protection")

            result: ProcessResult = self.processor.process(data)
            stages.extend(result.stages)
            if not result.ok:
                return finish(False, result.reason, result.message)
            png = result.face_png_b64()

            t0 = time.perf_counter()
            probe = self.embedder(result.face)
            stages.append(Stage("embed", True, (time.perf_counter() - t0) * 1000, "512-d FaceNet"))

            names = self.names(conn)
            t0 = time.perf_counter()
            if mode == "verify":
                try:
                    matcher = open_user(conn, scheme, self.ks, claimed_user, self.cfg.embedding_dim,
                                        self.cfg.cancelable_bits)
                except KeyError:
                    stages.append(Stage("decrypt", False, (time.perf_counter() - t0) * 1000,
                                        "no such user"))
                    return finish(False, "unknown_user", "no enrolled user with that id",
                                  face_png_b64=png)
                except TemplateIntegrityError as exc:
                    stages.append(Stage("decrypt", False, (time.perf_counter() - t0) * 1000, str(exc)))
                    return finish(False, "template_integrity_failure",
                                  "stored template failed its integrity check", face_png_b64=png)
            else:
                matcher = open_gallery(conn, scheme, self.ks, self.cfg.embedding_dim,
                                       self.cfg.cancelable_bits)
            stages.append(Stage("decrypt", True, (time.perf_counter() - t0) * 1000,
                                f"{len(matcher)} template(s) opened"))

            t0 = time.perf_counter()
            d = verify(matcher, probe, claimed_user, thr) if mode == "verify" \
                else identify(matcher, probe, thr)
            stages.append(Stage("match", True, (time.perf_counter() - t0) * 1000,
                                f"score {d.score:.3f}" if d.score is not None else d.reason))
            top = [{"user_id": u, "name": names.get(u, u), "score": round(s, 4)} for u, s in d.top]
            uid = d.user_id
            msg = ("match" if d.accepted else
                   "not enrolled: best score is below the threshold" if mode == "identify"
                   else "claim rejected: score below the threshold")
            return finish(d.accepted, d.reason, msg, score=d.score, user_id=uid,
                          user_name=names.get(uid) if uid else None, top=top, face_png_b64=png)
        finally:
            conn.close()

    # -- enrol / revoke / delete -----------------------------------------------------
    def _template_from_images(self, images: Sequence[bytes]) -> tuple:
        faces, rejected = [], []
        for i, data in enumerate(images):
            res = self.processor.process(data)
            if res.ok:
                faces.append(res.face)
            else:
                rejected.append({"index": i, "reason": res.reason, "message": res.message})
        if not faces:
            raise ServiceError("no usable face in the uploaded images: "
                               + "; ".join(f"#{r['index']} {r['reason']}" for r in rejected))
        emb = np.stack([self.embedder(f) for f in faces])
        return _mean_unit(emb), len(faces), rejected

    def enroll(self, images: Sequence[bytes], name: str) -> Dict:
        name = (name or "").strip()
        if not name or not slugify(name):
            raise ServiceError("a name is required")
        vector, n_used, rejected = self._template_from_images(images)
        conn = self.connect()
        base, k = f"member_{slugify(name)}", 1
        user_id = base
        while db.get_user(conn, user_id):
            k += 1
            user_id = f"{base}_{k}"
        enroll_users(conn, self.ks, [user_id], vector[None], self.cfg.cancelable_bits,
                     source="member", names={user_id: name})
        conn.close()
        return {"user_id": user_id, "name": name, "images_used": n_used, "rejected": rejected}

    def revoke(self, user_id: str, images: Sequence[bytes]) -> Dict:
        conn = self.connect()
        if db.get_user(conn, user_id) is None:
            conn.close()
            raise ServiceError("unknown user", 404)
        vector, n_used, rejected = self._template_from_images(images)
        version = reissue_user(conn, self.ks, user_id, vector, self.cfg.cancelable_bits)
        conn.close()
        return {"user_id": user_id, "key_version": version, "images_used": n_used,
                "rejected": rejected}

    def delete_user(self, user_id: str) -> None:
        conn = self.connect()
        user = db.get_user(conn, user_id)
        if user is None:
            conn.close()
            raise ServiceError("unknown user", 404)
        if user["source"] != "member":
            conn.close()
            raise ServiceError("only enrolled members can be deleted", 403)
        db.delete_user(conn, user_id)
        conn.commit()
        conn.close()

    def list_users(self) -> List[Dict]:
        conn = self.connect()
        users = db.list_users(conn)
        conn.close()
        return users

    def recent_log(self, limit: int = 50) -> List[Dict]:
        conn = self.connect()
        rows = db.recent_auth_log(conn, limit)
        conn.close()
        return rows

    # -- live threat simulation ------------------------------------------------------
    def run_threat(self, scenarios: Optional[List[str]], n_victims: int) -> Dict:
        from src.Security_Simulation import run_all
        from src.Threat_Scenario_Setup import build_lab
        with self._lab_lock:
            if self._lab is None or len(self._lab.victims) != n_victims:
                try:
                    self._lab = build_lab(self.paths, self.cfg, n_victims=n_victims)
                except FileNotFoundError as exc:
                    raise ServiceError(f"live simulation needs the Colab artifacts: {exc}", 409)
            return run_all(self._lab, scenarios)


def create_app(services: Optional[Services] = None):
    from flask import Flask, jsonify, request

    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
    svc = services or Services()
    app.extensions["services"] = svc

    @app.errorhandler(ServiceError)
    def _service_error(exc):
        return jsonify({"error": str(exc)}), exc.status

    @app.errorhandler(413)
    def _too_large(_exc):
        return jsonify({"error": "upload too large"}), 413

    def _image_bytes(field: str = "image") -> bytes:
        f = request.files.get(field)
        if f is None:
            raise ServiceError(f"file field '{field}' is required")
        return f.read()

    @app.get("/api/status")
    def status():
        artifacts = svc.paths
        users = svc.list_users()
        return jsonify({
            "schemes": list(SCHEMES), "calibrated": svc.calibrated(),
            "thresholds": svc.thresholds(), "n_users": len(users),
            "n_members": sum(1 for u in users if u["source"] == "member"),
            "results": {"evaluation": artifacts.file("eval_results.json").exists(),
                        "security": artifacts.file("security_results.json").exists()},
            "live_threat_available": artifacts.file("embeddings.npy").exists()
            and artifacts.file("splits.csv").exists(),
            "processor": {"min_face_px": svc.processor.min_face_px,
                          "min_sharpness": svc.processor.min_sharpness,
                          "min_confidence": svc.processor.min_confidence}})

    @app.get("/api/users")
    def users():
        return jsonify(svc.list_users())

    @app.get("/api/log")
    def log():
        return jsonify(svc.recent_log(int(request.args.get("limit", 50))))

    @app.post("/api/authenticate")
    def authenticate():
        form = request.form
        try:
            issued = float(form["issued_at"]) if form.get("issued_at") else None
        except ValueError as exc:
            raise ServiceError("issued_at must be a number of seconds since the epoch") from exc
        return jsonify(svc.authenticate(
            _image_bytes(), form.get("scheme", "aes"), form.get("mode", "identify"),
            form.get("claimed_user") or None, form.get("nonce") or None, issued))

    @app.post("/api/enroll")
    def enroll():
        if request.form.get("consent") != "true":
            raise ServiceError("the person in the photos must consent (consent=true)")
        images = [f.read() for f in request.files.getlist("images")]
        if not images:
            raise ServiceError("upload at least one photo in 'images'")
        return jsonify(svc.enroll(images, request.form.get("name", "")))

    @app.post("/api/revoke")
    def revoke():
        images = [f.read() for f in request.files.getlist("images")]
        if not images:
            raise ServiceError("upload a fresh photo in 'images'")
        return jsonify(svc.revoke(request.form.get("user_id", ""), images))

    @app.delete("/api/users/<user_id>")
    def delete_user(user_id: str):
        svc.delete_user(user_id)
        return jsonify({"deleted": user_id})

    _register_results_and_dashboard(app, svc)
    return app


def _register_results_and_dashboard(app, svc: Services) -> None:
    from dataclasses import asdict
    from flask import jsonify, request, send_from_directory
    from src.Threat_Scenario_Setup import SCENARIOS

    def _result_file(name: str, hint: str):
        f = svc.paths.file(name)
        if not f.exists():
            raise ServiceError(f"{name} not found in {svc.paths.artifacts}. {hint}", 404)
        return jsonify(read_json(f))

    @app.get("/api/results/evaluation")
    def evaluation():
        return _result_file("eval_results.json", "Copy it from the Colab run (Testing_Evaluation).")

    @app.get("/api/results/security")
    def security():
        return _result_file("security_results.json",
                            "Copy it from the Colab run (Security_Simulation).")

    @app.get("/api/threat/scenarios")
    def scenarios():
        return jsonify([asdict(s) for s in SCENARIOS.values()])

    @app.post("/api/threat/run")
    def threat_run():
        body = request.get_json(silent=True) or {}
        chosen = body.get("scenarios") or None
        if chosen is not None and not set(chosen) <= set(SCENARIOS):
            raise ServiceError(f"unknown scenarios; choose from {sorted(SCENARIOS)}")
        n = int(body.get("n_victims", 100))
        if not 20 <= n <= 500:
            raise ServiceError("n_victims must be between 20 and 500")
        return jsonify(svc.run_threat(chosen, n))

    @app.get("/", defaults={"path": ""})
    @app.get("/<path:path>")
    def dashboard(path: str):
        if path.startswith("api/"):
            return jsonify({"error": "not found"}), 404
        if not DASHBOARD_DIST.exists():
            return jsonify({"message": "API is running. Build the dashboard once with: "
                                       "cd frontend && npm install && npm run build"})
        if path and (DASHBOARD_DIST / path).is_file():
            return send_from_directory(DASHBOARD_DIST, path)
        return send_from_directory(DASHBOARD_DIST, "index.html")


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Demo backend")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-demo", help="enrol CelebA identities and calibrate thresholds")
    b.add_argument("--celeba-dir", type=Path, help="folder with the aligned CelebA .jpg files")
    b.add_argument("--identities", type=int, default=500)
    b.add_argument("--unknown", type=int, default=400, help="unknown images used for calibration")
    b.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("serve", help="start the API and dashboard")
    s.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1: photos are sensitive")
    s.add_argument("--port", type=int, default=5000)
    args = p.parse_args(argv)

    paths, cfg = get_paths().ensure(), load_pipeline_config()
    if args.command == "build-demo":
        folder = args.celeba_dir or images_dir(paths)
        if not Path(folder).is_dir():
            raise SystemExit(f"CelebA folder {folder} not found; pass --celeba-dir.")
        report = build_demo(paths, cfg, Path(folder), args.identities, args.unknown, args.seed)
        print(f"Demo gallery: {report['n_users']} users; rejections: {report['gallery_rejections']}")
        for scheme, t in report["schemes"].items():
            print(f"  {scheme:<11} verify>={t['verify']:.3f}  identify>={t['identify']:.3f}")
        return 0
    app = create_app()
    print(f"Dashboard + API on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
