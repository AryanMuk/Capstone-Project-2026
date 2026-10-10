from __future__ import annotations

import io
import os
import time

import numpy as np
import pytest
from PIL import Image

from common import cancelable, db
from common.artifacts import load_embeddings, load_splits, user_id_for
from common.config import SCHEMES, PipelineConfig, get_paths
from common.crypto import KeyStore, TemplateIntegrityError
from common.schemes import open_blob, protect
from tests.synthetic_data import make_fake_celeba, write_synthetic_artifacts

_ENV_VARS = ("CAPSTONE_WORK_DIR", "CAPSTONE_ARTIFACTS_DIR", "CAPSTONE_MASTER_KEY_HEX",
             "CAPSTONE_IMAGES_DIR", "CAPSTONE_DRIVE_DIR")


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Fresh, empty workspace for one test."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CAPSTONE_WORK_DIR", str(tmp_path / "work"))
    return get_paths().ensure()


@pytest.fixture(scope="module")
def synthetic_env(tmp_path_factory):
    """Workspace holding synthetic artifacts + the encrypted template database (shared by M3-M6)."""
    from src.Embedding_Encryption import build_database
    mp = pytest.MonkeyPatch()
    for var in _ENV_VARS:
        mp.delenv(var, raising=False)
    mp.setenv("CAPSTONE_WORK_DIR", str(tmp_path_factory.mktemp("synthetic") / "work"))
    paths = get_paths().ensure()
    splits, emb, cfg = write_synthetic_artifacts(paths, n_identities=120, noise=2.0, seed=3)
    report = build_database(cfg, paths)
    yield {"paths": paths, "splits": splits, "emb": emb, "cfg": cfg, "report": report}
    mp.undo()

class TestM1Data:
    def test_collect_zip_and_split(self, isolated, tmp_path):
        from src.Data_Collection import collect
        from src.Data_Preprocessing import run, validate_splits
        zip_path = make_fake_celeba(tmp_path / "fake", n_identities=40, make_zip=True)
        report = collect(zip_path, None, sample=30)
        assert report["n_identities"] == 40 and not report["checks"]["corrupt"]
        cfg = PipelineConfig(subset_identities=30)
        summary = run(cfg, check_images=5)
        assert (summary["known_identities"], summary["unknown_identities"]) == (24, 6)
        assert validate_splits(load_splits(), cfg) == []

    def test_split_reproducible_and_leak_free(self, synthetic_env):
        from src.Data_Preprocessing import assign_splits
        splits, cfg = synthetic_env["splits"], synthetic_env["cfg"]
        manifest = splits[["image", "identity"]]
        again = assign_splits(manifest, cfg)
        assert again.equals(splits)
        known = set(splits[splits.group == "known"].identity)
        unknown = set(splits[splits.group == "unknown"].identity)
        assert not known & unknown
        val_u = set(splits[(splits.group == "unknown") & (splits.partition == "val")].identity)
        test_u = set(splits[(splits.group == "unknown") & (splits.partition == "test")].identity)
        assert not val_u & test_u

    def test_crop_shape(self):
        from src.Data_Preprocessing import crop_celeba
        out = crop_celeba(Image.new("RGB", (178, 218)), 148, 160)
        assert out.size == (160, 160)


@pytest.mark.slow
class TestM2Embeddings:
    def test_unit_norm_and_resume(self, isolated, tmp_path):
        from src.Data_Collection import collect
        from src.Data_Preprocessing import run
        from src.Embedding_Generation import generate, get_device, load_facenet
        try:
            load_facenet(get_device("cpu"))
        except Exception as exc:  
            pytest.skip(f"FaceNet weights unavailable: {exc}")
        collect(None, make_fake_celeba(tmp_path / "fake", n_identities=20), sample=10)
        cfg = PipelineConfig(subset_identities=20)
        run(cfg, check_images=0)
        meta = generate(cfg, shard_size=32, device=get_device("cpu"), num_workers=0, limit=64)
        emb = np.load(get_paths().file("embeddings_limit.npy"))
        assert emb.shape == (64, 512)
        assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)
        again = generate(cfg, shard_size=32, device=get_device("cpu"), num_workers=0, limit=64)
        assert again["shards_resumed"] == 2 and again["images_embedded_this_run"] == 0


class TestM3Crypto:
    @pytest.fixture()
    def vec(self):
        v = np.random.default_rng(0).standard_normal(512).astype(np.float32)
        return v / np.linalg.norm(v)

    @pytest.mark.parametrize("scheme", ["aes", "ecies", "cancelable"])
    def test_tamper_and_owner_checks(self, scheme, vec):
        ks = KeyStore(os.urandom(32))
        blob = protect(scheme, ks, "id_00001", 1, vec, 256)
        open_blob(scheme, ks, "id_00001", 1, blob, 512, 256)                  
        with pytest.raises(TemplateIntegrityError):
            open_blob(scheme, ks, "id_00002", 1, blob, 512, 256)             
        bad = bytearray(blob)
        bad[len(bad) // 2] ^= 1
        with pytest.raises(TemplateIntegrityError):
            open_blob(scheme, ks, "id_00001", 1, bytes(bad), 512, 256)        
        with pytest.raises(TemplateIntegrityError):
            open_blob(scheme, ks, "id_00001", 2, blob, 512, 256)            

    @pytest.mark.parametrize("scheme", ["plaintext", "aes", "ecies"])
    def test_float_schemes_round_trip_exactly(self, scheme, vec):
        ks = KeyStore(os.urandom(32))
        blob = protect(scheme, ks, "u", 1, vec, 256)
        assert np.array_equal(open_blob(scheme, ks, "u", 1, blob, 512, 256), vec)

    def test_keys_are_deterministic_and_version_changes_code(self, vec):
        master = os.urandom(32)
        a, b = KeyStore(master), KeyStore(master)
        assert np.array_equal(a.cancelable_projection(256, 512), b.cancelable_projection(256, 512))
        proj = a.cancelable_projection(256, 512)
        c1 = cancelable.binarize(vec, a.cancelable_signs("u", 1, 512), proj)
        c1b = cancelable.binarize(vec, b.cancelable_signs("u", 1, 512), proj)
        assert np.array_equal(c1, c1b)
        # Re-keyed codes of the same face are unrelated ON AVERAGE (any single pair keeps a small
        # random similarity because the two sign-flipped inputs are slightly correlated).
        scores = [cancelable.code_score(c1, cancelable.binarize(
            vec, a.cancelable_signs("u", v, 512), proj)) for v in range(2, 42)]
        assert abs(np.mean(scores)) < 0.08 and max(abs(x) for x in scores) < 0.5

    def test_database_built_and_verified(self, synthetic_env):
        report = synthetic_env["report"]
        assert report["n_users"] == 96
        sizes = {s: report["storage"][s]["mean_bytes"] for s in SCHEMES}
        assert sizes["cancelable"] < sizes["plaintext"] < sizes["aes"] < sizes["ecies"]
        for scheme in ("aes", "ecies", "cancelable"):
            v = report["verification"][scheme]
            assert v["tamper_detected"] == v["checked"]


class TestM4Matching:
    @pytest.fixture()
    def matchers(self, synthetic_env):
        from src.Query_Authorisation import open_gallery
        from src.Embedding_Encryption import TEMPLATES_DB
        paths, cfg = synthetic_env["paths"], synthetic_env["cfg"]
        conn = db.connect(paths.file(TEMPLATES_DB))
        ks = KeyStore.load_or_create(paths, create=False)
        yield {s: open_gallery(conn, s, ks, cfg.embedding_dim, cfg.cancelable_bits) for s in SCHEMES}
        conn.close()

    def test_float_schemes_score_identically(self, synthetic_env, matchers):
        from src.Query_Authorisation import score_matrix
        probes = synthetic_env["emb"][:30]
        base = score_matrix(matchers["plaintext"], probes)
        for s in ("aes", "ecies"):
            assert np.array_equal(score_matrix(matchers[s], probes), base)

    def test_cancelable_vectorised_equals_reference(self, synthetic_env, matchers):
        from src.Query_Authorisation import score_matrix
        m, probes = matchers["cancelable"], synthetic_env["emb"][:4]
        got = score_matrix(m, probes)
        ref = np.array([[cancelable.code_score(cancelable.binarize(x, m.signs[u], m.proj),
                                               m.gallery[u]) for u in range(len(m))]
                        for x in probes])
        assert np.allclose(got, ref, atol=1e-6)

    def test_identify_accepts_known_and_rejects_unknown(self, synthetic_env, matchers):
        from src.Query_Authorisation import identify
        splits, emb = synthetic_env["splits"], None
        paths = synthetic_env["paths"]
        emb, splits = load_embeddings(paths)
        test = splits[splits.partition == "test"]
        known = test[test.group == "known"].iloc[0]
        unknown = test[test.group == "unknown"].iloc[0]
        d1 = identify(matchers["aes"], emb[known["row"]], threshold=0.3)
        d2 = identify(matchers["aes"], emb[unknown["row"]], threshold=0.3)
        assert d1.accepted and d1.user_id == user_id_for(known["identity"])
        assert not d2.accepted and d2.reason == "below_threshold"

    def test_unknown_claim_and_replay_protection(self, synthetic_env, matchers):
        from src.Query_Authorisation import check_request_freshness, verify
        assert verify(matchers["aes"], np.ones(512, np.float32), "nobody", 0.5).reason == "unknown_user"
        conn = db.connect(":memory:")
        db.init_schema(conn)
        assert check_request_freshness(conn, "n1", time.time()) == (True, "ok")
        assert check_request_freshness(conn, "n1", time.time()) == (False, "replayed_nonce")
        assert check_request_freshness(conn, "n2", time.time() - 3600) == (False, "stale_request")


class TestM5Evaluation:
    def test_ties_count_against_the_system(self):
        from src.Testing_Evaluation import ranks_of_truth
        scores = np.array([[0.5, 0.5, 0.1], [0.9, 0.2, 0.1]])
        assert ranks_of_truth(scores, np.array([0, 0])).tolist() == [2, 1]

    def test_eer_matches_theory(self):
        from src.Testing_Evaluation import verification_metrics
        rng = np.random.default_rng(0)
        m = verification_metrics(rng.normal(2, 1, 50000), rng.normal(0, 1, 50000))
        assert abs(m["eer"] - 0.1587) < 0.01        
        assert 0.91 < m["auc"] < 0.93

    def test_impostor_sampling_never_uses_true_column(self):
        from src.Testing_Evaluation import sample_impostor_scores
        scores = np.zeros((200, 5))
        true_idx = np.random.default_rng(1).integers(0, 5, 200)
        scores[np.arange(200), true_idx] = 99.0
        out = sample_impostor_scores(scores, true_idx, 10, np.random.default_rng(2))
        assert out.max() == 0.0

    def test_open_set_report(self):
        from src.Testing_Evaluation import open_set_report
        rep = open_set_report(np.full(10, 0.9), np.full(10, 0.9), np.ones(10, bool),
                              np.linspace(0, 0.5, 100), np.linspace(0, 0.5, 100))
        assert rep["0.01"]["dir_test"] == 1.0 and rep["0.01"]["fpir_test"] <= 0.02

    def test_full_evaluation_on_synthetic(self, synthetic_env):
        from src.Testing_Evaluation import run_evaluation
        res = run_evaluation(synthetic_env["cfg"], synthetic_env["paths"], max_probes=2000,
                             figures=False)["schemes"]
        for s in ("aes", "ecies"):                   
            assert res[s]["closed_set"] == res["plaintext"]["closed_set"]
            assert res[s]["verification"]["auc"] == res["plaintext"]["verification"]["auc"]
        assert res["plaintext"]["closed_set"]["rank1"] > 0.9
        assert res["cancelable"]["efficiency"]["bytes_per_template"] < 100


class TestM6Security:
    def test_every_scenario_behaves_as_designed(self, synthetic_env):
        from src.Security_Simulation import run_all
        from src.Threat_Scenario_Setup import build_lab
        lab = build_lab(synthetic_env["paths"], synthetic_env["cfg"], n_victims=40,
                        workdir=synthetic_env["paths"].artifacts / "lab_test")
        r = run_all(lab)["scenarios"]
        theft = r["db_theft"]["per_scheme"]
        assert theft["plaintext"]["recoverable_fraction"] == 1.0
        assert all(theft[s]["recoverable_fraction"] == 0.0 for s in ("aes", "ecies", "cancelable"))
        key = r["key_compromise"]["per_scheme"]
        assert key["plaintext"]["blast_radius_single_user_key"] == 1.0
        assert key["aes"]["blast_radius_single_user_key"] <= 0.05
        assert all(key[s]["blast_radius_master_key"] == 1.0 for s in SCHEMES)
        tamper = r["tampering"]["per_scheme"]
        assert all(tamper[s]["detection_rate_bitflip"] == 1.0 for s in ("aes", "ecies", "cancelable"))
        assert tamper["plaintext"]["asr_substitution"] == 1.0
        replay = r["replay"]["per_scheme"]
        assert all(replay[s]["replay_accept_rate_protected"] == 0.0 for s in SCHEMES)
        rev = r["revocation"]["per_scheme"]
        assert rev["aes"]["asr_old_credential_after_reissue"] >= 0.8
        assert rev["cancelable"]["asr_old_credential_before_reissue"] == 1.0
        assert rev["cancelable"]["asr_old_credential_after_reissue"] <= 0.2
        link = r["unlinkability"]["per_scheme"]
        assert link["plaintext"]["linkage_auc_decrypted"] > 0.95
        assert link["cancelable"]["linkage_auc_decrypted"] < 0.65


class TestM7ImageProcessor:
    """Stages that reject BEFORE face detection need no model, so they run anywhere."""

    @staticmethod
    def _png(size):
        buf = io.BytesIO()
        Image.new("RGB", size, (120, 120, 120)).save(buf, "PNG")
        return buf.getvalue()

    def test_rejections_before_detection(self):
        from src.Data_Preprocessing import ImageProcessor
        proc = ImageProcessor()
        assert proc.process(b"definitely not an image").reason == "invalid_image"
        assert proc.process(self._png((30, 30))).reason == "image_too_small"
        assert proc.process(b"0" * 9_000_000).reason == "image_too_large"

    def test_sharpness_orders_blurry_below_sharp(self):
        from src.Data_Preprocessing import sharpness
        rng = np.random.default_rng(0)
        sharp = rng.integers(0, 255, (160, 160, 3), dtype=np.uint8)
        flat = np.full((160, 160, 3), 128, dtype=np.uint8)
        assert sharpness(sharp) > 1000 and sharpness(flat) == 0.0


class _StubProcessor:
    """Stands in for MTCNN: the 'photo' is just b'<anything>:<label>'; b'noface...' is rejected."""
    min_face_px, min_sharpness, min_confidence = 60, 20.0, 0.9

    def process(self, data):
        from src.Data_Preprocessing import ProcessResult, Stage
        if data.startswith(b"noface"):
            return ProcessResult(False, "no_face", "no face detected",
                                 [Stage("detect", False, 0.1, "no face detected")])
        import zlib
        seed = zlib.crc32(data.split(b":", 1)[1]) % 250 + 1
        return ProcessResult(True, None, "ok", [Stage("validate", True, 0.1, "stub")],
                             np.full((160, 160, 3), seed, np.uint8), [0.0, 0.0, 1.0, 1.0])


class _StubEmbedder:
    """Same label -> same identity centre plus per-call noise (a genuine-looking variation)."""

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def __call__(self, face):
        centre = np.random.default_rng(int(face[0, 0, 0])).standard_normal(512)
        v = centre / np.linalg.norm(centre) + 0.9 / np.sqrt(512) * self.rng.standard_normal(512)
        return (v / np.linalg.norm(v)).astype(np.float32)


class TestM7Api:
    @pytest.fixture()
    def client(self, isolated, tmp_path):
        from src.Backend import Services, create_app
        svc = Services(paths=isolated, cfg=PipelineConfig(), processor=_StubProcessor(),
                       embedder=_StubEmbedder(), keystore=KeyStore(os.urandom(32)),
                       db_path=tmp_path / "demo.sqlite")
        app = create_app(svc)
        app.testing = True
        return app.test_client(), svc

    @staticmethod
    def _form(label, **extra):
        return {"image": (io.BytesIO(b"x:" + label.encode()), "p.jpg"), **extra}

    def _enroll(self, c, name="Alice", label="alice", consent="true"):
        files = [(io.BytesIO(b"x:" + label.encode()), f"{i}.jpg") for i in range(3)]
        return c.post("/api/enroll", data={"name": name, "consent": consent, "images": files},
                      content_type="multipart/form-data")

    def test_enrol_requires_consent(self, client):
        c, _ = client
        assert self._enroll(c, consent="false").status_code == 400
        assert self._enroll(c).get_json()["user_id"] == "member_alice"

    @pytest.mark.parametrize("scheme", list(SCHEMES))
    def test_identify_and_verify(self, client, scheme):
        c, _ = client
        self._enroll(c)
        post = lambda **kw: c.post("/api/authenticate", data=self._form(  
            kw.pop("label"), scheme=scheme, **kw), content_type="multipart/form-data").get_json()
        mine = post(label="alice", mode="identify")
        assert mine["accepted"] and mine["user_id"] == "member_alice"
        assert [s["name"] for s in mine["stages"]][-3:] == ["embed", "decrypt", "match"]
        stranger = post(label="bob", mode="identify")
        assert not stranger["accepted"] and stranger["reason"] == "below_threshold"
        assert post(label="alice", mode="verify", claimed_user="member_alice")["accepted"]
        assert not post(label="bob", mode="verify", claimed_user="member_alice")["accepted"]
        assert post(label="alice", mode="verify", claimed_user="ghost")["reason"] == "unknown_user"

    def test_rejections_replay_and_validation(self, client):
        c, _ = client
        self._enroll(c)
        rej = c.post("/api/authenticate", data=self._form("x", scheme="aes", mode="identify")
                     | {"image": (io.BytesIO(b"noface"), "n.jpg")},
                     content_type="multipart/form-data").get_json()
        assert not rej["accepted"] and rej["reason"] == "no_face"
        body = lambda: {**self._form("alice"), "scheme": "aes", "mode": "identify",  
                        "nonce": "same-nonce", "issued_at": str(time.time())}
        first = c.post("/api/authenticate", data=body(), content_type="multipart/form-data")
        replay = c.post("/api/authenticate", data=body(), content_type="multipart/form-data")
        assert first.get_json()["accepted"]
        assert replay.get_json()["reason"] == "replayed_nonce"
        bad = c.post("/api/authenticate", data=self._form("alice", scheme="rot13", mode="identify"),
                     content_type="multipart/form-data")
        assert bad.status_code == 400

    def test_revoke_then_delete(self, client):
        c, svc = client
        self._enroll(c)
        files = [(io.BytesIO(b"x:alice"), "new.jpg")]
        out = c.post("/api/revoke", data={"user_id": "member_alice", "images": files},
                     content_type="multipart/form-data").get_json()
        assert out["key_version"] == 2
        for scheme in SCHEMES:
            r = c.post("/api/authenticate", data=self._form("alice", scheme=scheme, mode="identify"),
                       content_type="multipart/form-data").get_json()
            assert r["accepted"], scheme
        assert c.delete("/api/users/member_alice").status_code == 200
        assert c.get("/api/users").get_json() == []

    def test_celeba_users_cannot_be_deleted_and_missing_results_404(self, client):
        from src.Embedding_Encryption import enroll_users
        c, svc = client
        conn = svc.connect()
        enroll_users(conn, svc.ks, ["id_00001"], np.eye(512, dtype=np.float32)[:1], 256)
        conn.close()
        assert c.delete("/api/users/id_00001").status_code == 403
        assert c.get("/api/results/evaluation").status_code == 404
        assert c.get("/api/status").get_json()["calibrated"] is False
        assert c.get("/api/nonexistent").status_code == 404

    def test_live_threat_run(self, client, synthetic_env):
        from src.Backend import Services, create_app
        c, svc = client
        assert c.post("/api/threat/run", json={"n_victims": 30}).status_code == 409  
        live = Services(paths=synthetic_env["paths"], cfg=synthetic_env["cfg"],
                        processor=_StubProcessor(), embedder=_StubEmbedder(),
                        keystore=KeyStore(os.urandom(32)),
                        db_path=synthetic_env["paths"].artifacts / "api_test.sqlite")
        app = create_app(live)
        out = app.test_client().post("/api/threat/run",
                                     json={"scenarios": ["db_theft", "tampering"], "n_victims": 30})
        body = out.get_json()
        assert out.status_code == 200 and set(body["scenarios"]) == {"db_theft", "tampering"}
        assert body["scenarios"]["db_theft"]["per_scheme"]["plaintext"]["recoverable_fraction"] == 1.0
