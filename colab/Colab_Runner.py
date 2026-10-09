# %% Cell 1 — Drive and repo
from google.colab import drive

drive.mount("/content/drive")

REPO_URL = "https://github.com/AryanMuk/Capstone-Project-2026.git"   
# %cd /content
# !git clone $REPO_URL capstone || (cd capstone && git pull)
# %cd /content/capstone

# %% Cell 2 — dependencies (facenet-pytorch pins old torch/numpy, so it is installed --no-deps)
# !pip install -q -r requirements-colab.txt
# !pip install -q --no-deps facenet-pytorch==2.6.0
# !python -m src.Project_Setup check
# !python -m src.Project_Setup init
# !python -m src.Project_Setup drive-check     

# %% Cell 3 — smoke run first (about 2,000 identities), then the full run
SUBSET = 2000
ZIP = "/content/drive/MyDrive/capstone/celeba.zip"
# !python -m src.Integration run --zip $ZIP --subset $SUBSET --fp16

# %% Cell 4 — package for the laptop (written into Drive next to the artifacts)
# !python -m src.Integration bundle
# Download the folder  MyDrive/capstone/artifacts/laptop_bundle  and copy its files into
# work\artifacts on the laptop. The master key stays in Drive; the laptop builds its own.
