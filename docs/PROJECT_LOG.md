# Project Log — Evaluation of Biometric Template Protection for CNN-Based Facial Authentication

Structured to CMMI for Development, Maturity Level 3

## 1. Document Control

| Item                | Value                                 |
| ------------------- | ------------------------------------- |
| Team                | Suvrat (lead), Aryan, Jaskaran, Ayati |
| Plan window (Gantt) | 21 Sep – 15 Oct 2026                  |

## 2. Process-Area Coverage (Maturity Level 2 + 3 Areas Used)

| Maturity Level 2 Areas                                                                                                                                                                                                                                                                                                                                                                        | Maturity Level 3 Areas                                                                                                                                                                                                                                                                                                                                                                                                                     |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| • [REQM Requirements Management](#3-requirements-register)<br>• [PP Project Planning](#4-project-plan-versus-gantt)<br>• [PMC Project Monitoring & Control](#9-activity-log)<br>• [MA Measurement & Analysis](#10-measurement-and-analysis-ma)<br>• [PPQA Process & Product Quality Assurance](#8-quality-assurance-ppqa)<br>• [CM Configuration Management](#11-configuration-management-cm) | • [RD Requirements Development](#3-requirements-register)<br>• [TS Technical Solution](#5-decision-log-dar)<br>• [PI Product Integration](../src/Integration_Suvrat.py)<br>• [VER Verification / VAL Validation](#7-verification-and-validation-record)<br>• [RSKM Risk Management](#6-risk-register-rskm)<br>• [DAR Decision Analysis & Resolution](#5-decision-log-dar)<br>• [OPD Organizational Process Definition](../CONTRIBUTING.md) |

## 3. Requirements Register

| ID  | Requirement                                                                          | Realised in                                                      |
| --- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| R1  | Compare plaintext, AES, ECIES and a non-invertible cancelable template               | `common/schemes.py`, `common/crypto.py`, `common/cancelable.py`  |
| R2  | Closed-set, verification and open-set evaluation; thresholds on val, results on test | `Testing_Evaluation_Ayati.py`                                    |
| R3  | Theft scenarios with metrics (six)                                                   | `Threat_Scenario_Setup_Aryan.py`, `Security_Simulation_Aryan.py` |
| R4  | Dashboard: authentication demo, theft metrics, scheme comparison                     | `Backend_Aryan.py`, `frontend/`                                  |
| R5  | Image processor rejects a teammate who is not enrolled                               | `ImageProcessor`, `Services.authenticate`                        |
| R6  | SQLite database                                                                      | `common/db.py`                                                   |
| R7  | Colab-compliant, `SUBSET` switch, `.py` files                                        | `colab/`, `Integration_Suvrat.py`, `--subset`                    |
| R8  | Separate virtual environment (Windows 11, Python 3.11, CPU)                          | `setup_env.ps1`, `requirements-laptop.txt`                       |
| R9  | Model FaceNet (InceptionResnetV1, vggface2)                                          | `Embedding_Generation_Suvrat.py`                                 |
| R10 | CMMI L3 documentation                                                                | this file                                                        |

## 4. Project Plan Versus Gantt

| Gantt task (planned)                       | Owner    | Module                                  |
| ------------------------------------------ | -------- | --------------------------------------- |
| Project Initiation & Staffing, Tool SignUp | Suvrat   | `Project_Setup_Suvrat.py`               |
| Data Collection (23 Sep)                   | Jaskaran | `Data_Collection_Jaskaran.py`           |
| Data Preprocessing (24 Sep)                | Aryan    | `Data_Preprocessing_Aryan.py`           |
| Embedding Generation (25–28 Sep)           | Suvrat   | `Embedding_Generation_Suvrat.py`        |
| Embedding Encryption (29–30 Sep)           | Suvrat   | `Embedding_Encryption_Suvrat.py`        |
| Query Authorisation (1–2 Oct)              | Jaskaran | `Query_Authorisation_Jaskaran.py`       |
| Threat Scenario Setup (5 Oct)              | Aryan    | `Threat_Scenario_Setup_Aryan.py`        |
| Testing & Evaluation (6 Oct)               | Ayati    | `Testing_Evaluation_Ayati.py`, `tests/` |
| Security Simulation (7 Oct)                | Aryan    | `Security_Simulation_Aryan.py`          |
| Backend (8–9 Oct)                          | Aryan    | `Backend_Aryan.py`                      |
| Frontend                                   | Jaskaran | `frontend/`                             |
| Integration (13–14 Oct)                    | Suvrat   | `Integration_Suvrat.py`, `colab/`       |

Schedule note: the code for all tasks was produced on 2026-10-04, so the earlier Gantt dates were not met as
planned dates of work. The team should record this honestly in its PMC review and must commit on the real day.

## 5. Decision Log (DAR)

| ID      | Decision                                                                          | Alternatives               | Rationale                                                                         |
| ------- | --------------------------------------------------------------------------------- | -------------------------- | --------------------------------------------------------------------------------- |
| DEC-001 | FaceNet (facenet-pytorch, vggface2)                                               | ArcFace                    | Simpler install on Colab and CPU laptop; ArcFace was considered then rolled back. |
| DEC-002 | SQLite                                                                            | server database            | Single-file, runs on the laptop with no service.                                  |
| DEC-003 | Cancelable scheme must be non-invertible: sign(P·(s⊙x)), k=256 bits               | invertible random rotation | Stored code cannot be inverted to the embedding; costs accuracy (reported).       |
| DEC-004 | ECIES composed from X25519 + HKDF + AES-GCM                                       | RSA                        | Library primitives, small keys; no custom cipher.                                 |
| DEC-005 | Known/unknown identity split 80/20, 5 gallery images, val/test probes, seed 42    | random image split         | Allows closed-set, verification and open-set from one protocol.                   |
| DEC-006 | User template = L2-normalised mean of 5 gallery embeddings                        | per-image templates        | One template per user keeps storage and matching simple.                          |
| DEC-007 | CelebA centre crop 148→160 and MTCNN margin 0.15                                  | other crops                | **Assumption, untuned.**                                                          |
| DEC-008 | Install facenet-pytorch with `--no-deps` on Colab                                 | follow its pins            | Its pins (torch<2.3, numpy<2) conflict with Colab.                                |
| DEC-009 | Docker deferred                                                                   | Docker image               | Not needed yet; venv suffices.                                                    |
| DEC-010 | Python 3.11 venv on laptop                                                        | 3.14 (default)             | Pinned wheels exist for 3.11.                                                     |
| DEC-011 | Shared secret projection P plus per-user sign flips derived by HKDF + key_version | per-user P                 | Deterministic, revocable via key_version.                                         |
| DEC-012 | OpenCV and Seaborn not used                                                       | —                          | Not needed; fewer dependencies.                                                   |
| DEC-013 | Dashboard charts hand-written in SVG                                              | chart library              | No extra dependency; accessible table view.                                       |

## 6. Risk Register (RSKM)

| ID  | Risk                                                            | Mitigation                                           | Status                     |
| --- | --------------------------------------------------------------- | ---------------------------------------------------- | -------------------------- |
| RK1 | Colab dependency conflicts                                      | DEC-008, pinned `requirements-colab.txt`             | open until first Colab run |
| RK2 | Processor thresholds reject genuine members or accept strangers | tune on real team photos; report                     | open                       |
| RK3 | Cancelable accuracy loss                                        | report honestly; bits ablation (`--cancelable-bits`) | accepted                   |
| RK4 | Privacy of team photos                                          | in-memory processing, localhost, consent, delete     | mitigated                  |
| RK5 | Windows-only setup issues                                       | `setup_env.ps1` stops on first failing step          | open until run             |
| RK6 | Synthetic numbers mistaken for results                          | README and §7 state this                             | mitigated                  |

## 7. Verification And Validation Record

Done on 2026-10-04 in a Linux sandbox:

- `pytest -m "not slow"`: 32 passed; `pytest -m slow`: 1 passed (real FaceNet weights, real-model backend path).
- Full synthetic pipeline run: encrypt → evaluate → security simulation → API; cancelable torch scoring equals NumPy reference.
- Dashboard built with Vite and rendered in headless Chromium against the Flask server with synthetic artifacts: Compare and Theft views show data, no console errors.

**Not done:** run on Google Colab or an A100 (timings unknown); run on Windows 11; real CelebA accuracy; rejecting a real
photo of an unenrolled person; authenticate view with a real uploaded face in the browser; tuning of processor thresholds.

## 8. Quality Assurance (PPQA)

Checklist per PR: tests pass; `CODEBASE_MAP.md` regenerated; no duplicate helpers (reuse `common/`); no secrets or photos committed;
log updated. Review is by a teammate other than the author.

## 9. Activity Log

| Date          | Activity                                                                                         | By     |
| ------------- | ------------------------------------------------------------------------------------------------ | ------ |
| **[TO FILL]** | Project kickoff, roles, tool sign-ups                                                            | team   |
| **[TO FILL]** | Literature review completed                                                                      | team   |
| **[TO FILL]** | CelebA downloaded to laptop                                                                      | team   |
| 2026-10-04    | Requirements clarified (closed/open set, FaceNet, SQLite, non-invertible template, CMMI L3)      | Suvrat |
| 2026-10-04    | Pipeline, backend, dashboard, integration, Colab runner and documentation produced (AI-assisted) | team   |
| 2026-10-04    | Smoke tests and render check run (§7)                                                            | team   |

## 10. Measurement And Analysis (MA)

Measures collected by the pipeline: rank-1, AUC, EER, FRR at FAR 1%, DIR at FPIR 0.001/0.01/0.1, verify latency,
decrypt time per template, bytes per template, attack success rates per scenario, stage timings
(`integration_report.json`). Real values: **[TO FILL after the Colab run]**.
Findings to check on real data (seen on synthetic data only): the 256-bit cancelable template loses accuracy, especially for 1:N;
an embedding reconstructed from a stolen cancelable code still matches after re-keying; a leaked master key defeats AES/ECIES
templates; re-keyed codes of the same face keep a small random similarity.

## 11. Configuration Management (CM)

Git, per-member feature branches, pull requests to `main`. Baselines: tag after each verified milestone.
Generated/ignored: `work/`, `keys/`, `*.sqlite`, `*.npy`, `frontend/node_modules`, `frontend/dist`.
