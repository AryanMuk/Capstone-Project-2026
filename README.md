# Template Protection Lab

Evaluation of biometric template protection for CNN-based facial authentication (FaceNet 512-d embeddings).
Four schemes are compared on the same data: **plaintext** (control), **AES-256-GCM**, **ECIES (X25519)** and a
**cancelable, non-invertible** binary template. Heavy work runs on Google Colab (GPU); the demo dashboard runs on a
Windows 11 laptop (CPU only) against a SQLite database.

Every stage is one file named `<Task>.py` in `src/`. `CODEBASE_MAP.md` lists every public function; `docs/PROJECT_LOG.md` is the CMMI Level 3 record.

## 1. Prepare the dataset (once)

Place the `img_align_celeba` folder **together with** `identity_CelebA.txt` in `celebal_upload` folder and zip and rename it as `celeba.zip` and upload it to Google Drive at `MyDrive/capstone/celeba.zip`.

## 2. Run the pipeline on Colab

Clone the repo or download the zipped version of the code and unzip it. Then upload the codebase to Google Drive at `MyDrive/capstone_code/capstone`. Note that repo folder name will be different, so ensure to rename it to `capstone` on drive.

Open a new notebook at colab.research.google.com. Go to Runtime → Change runtime type and pick A100 GPU.

Follow the code cells in `Capstone_Run.ipynb`.
Note: `!python -m src.Integration run --zip /content/drive/MyDrive/capstone/celeba.zip --subset 2000 --fp16`, in this command subset flag is kept for trial runs. Removing it will run the full dataset.
Once pipeline completed, finish with `!python -m src.Integration bundle`. Artifacts persist in Drive (`MyDrive/capstone/artifacts`)
Note: Since different runs have different setttings, so before every run remove the previous artifacts using the last command in the `Capstone_Run.ipynb`

## 3. Set up the laptop (PowerShell, repo root)

Copy the six files from `laptop_bundle` into `work\artifacts`. Create the folder tree if not present.
Then:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_env.ps1
.\.venv\Scripts\Activate.ps1
cd frontend; npm install; npm run build; cd ..
```

```powershell
python -m src.Backend build-demo --celeba-dir "path_to_img_align_celeba"
python -m src.Backend serve
```

(`python -m src.Integration demo --celeba-dir ...` does both.)

Dashboard: **Authenticate** (upload a photo, see each processing stage and the decision, replay a request),
**Compare schemes** (ROC, CMC, open-set, latency, storage), **Theft scenarios** (six attacks, stored or live),
**People** (enrol team members with consent, re-issue keys, delete).
To show the rejection demo: enrol some teammates, leave one out, upload the teammate who is not enrolled.

## Tests

`python -m pytest -m "not slow"` runs the synthetic-data smoke tests; `-m slow` also downloads the FaceNet weights.

## Privacy

Photos are processed in memory and never stored; the server binds to `127.0.0.1`; members consent and can be
deleted. `keys/` and `*.sqlite` are git-ignored — never commit them.

## Read before quoting numbers

Smoke tests use synthetic embeddings: their accuracy figures are **not** results. Real accuracy, timings and the
GPU runtime come only from the Colab run on CelebA. Image-processor thresholds (face size, sharpness, margin) are
untuned defaults.
