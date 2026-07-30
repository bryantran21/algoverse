"""Central config. Every module imports from here so a single edit re-points the whole pipeline.

Swap MODEL_ID to move to Gemma-3; swap DATASET_* to move to another QA set.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
# Swap to "google/gemma-3-...-it" later. Keep it a plain HF repo id.
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
# For a single free Colab T4 (16 GB) the 7B bf16 model is a tight/OOM risk.
# Set MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct there, or LOAD_IN_4BIT=1.
TORCH_DTYPE = "bfloat16"
DEVICE_MAP = "auto"          # shards across 2x T4 on Kaggle
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"

# --------------------------------------------------------------------------- #
# Dataset (swappable loader — see src/inference.py:load_items)
# --------------------------------------------------------------------------- #
DATASET_NAME = os.environ.get("DATASET_NAME", "google/boolq")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "validation")
# Field names for the active dataset. Change these three when you swap datasets.
DATASET_FIELDS = {
    "passage": "passage",
    "question": "question",
    "answer": "answer",      # BoolQ: bool True/False
}
# Map raw ground-truth value -> canonical label string used for scoring.
def normalize_answer(raw) -> str:
    """BoolQ: bool -> 'yes'/'no'. Override when swapping to an extractive set."""
    if isinstance(raw, bool):
        return "yes" if raw else "no"
    return str(raw).strip().lower()

# Allowed answer labels for BoolQ (used to constrain / parse the prediction).
ANSWER_LABELS = ("yes", "no")

# --------------------------------------------------------------------------- #
# Rendering defaults (Typst)
# --------------------------------------------------------------------------- #
# Fixed page box -> image dimensions stay constant as font size changes, so the
# ONLY thing the sweep varies is text density (compression), not aspect ratio.
RENDER = {
    "page_width_mm": 160.0,
    "page_height_mm": 120.0,
    "margin_mm": 8.0,
    "ppi": 150,                 # pixels-per-inch when rasterizing to PNG
    "font": "DejaVu Sans",      # a font present on Kaggle/Colab
    "font_size_pt": 14.0,       # default; Phase 2 sweeps this
}
FONT_SWEEP = [22.0, 18.0, 14.0, 11.0, 9.0, 7.0]   # large->small = easy->hard
FONT_SWEEP_FAMILIES = ["DejaVu Sans", "DejaVu Serif"]

# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
SEED = 1234
MAX_NEW_TOKENS = 8            # answers are short; keep generation cheap
BATCH_SIZE = 4               # per-forward batch for behavioral runs
GREEDY = True                # do_sample=False, deterministic

# --------------------------------------------------------------------------- #
# Paths (data/ is gitignored; point DATA_DIR at Drive on Colab to persist)
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))
PREDS_DIR = DATA_DIR / "preds"
ACTS_DIR = DATA_DIR / "activations"
BUCKETS_DIR = DATA_DIR / "buckets"
INTERP_DIR = DATA_DIR / "interp"
FIGS_DIR = DATA_DIR / "figures"

for _d in (DATA_DIR, PREDS_DIR, ACTS_DIR, BUCKETS_DIR, INTERP_DIR, FIGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
