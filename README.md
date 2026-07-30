# Visualized-text modality gap in VLMs — mechanistic analysis

Self-contained research repo. Renders text QA to images, runs a VLM in
text / image / combined modes, captures per-layer activations, and runs
failure-conditioned interpretability (logit lens, activation patching, probing)
to locate **where** a VLM loses the ability to answer a question it can answer
from text once that text is shown as an image.

Everything runs on **free GPU tiers** (Kaggle 2×T4 / Colab free T4): inference +
a tiny logistic probe only. No paid GPU, no base-model training.

## Status
- **Phase 1 (infra): built.** Not yet run — needs a GPU (Kaggle/Colab).
- Phases 2–4: not started (built on demand after each go/no-go gate).

## Layout
```
config.py                 # MODEL_ID, dataset, render defaults, paths, seed — edit here to re-point everything
src/
  rendering.py            # text QA -> PNG via Typst (fixed page box; font size = compression)
  inference.py            # load model; load_items; run text/image/combined; preds + token counts; incremental CSV
  activations.py          # capture answer-position hidden state per layer (last-token only; storage-safe)
  failure_sets.py         # [Phase 3] bucket items into IMG_wrong/TEXT_correct etc.
  interp_logit_lens.py    # [Phase 4-A] project hidden states -> vocab; P(correct token) vs layer
  interp_patching.py      # [Phase 4-B] patch clean(text) -> corrupted(image) at answer position, per layer
  interp_probe.py         # [Phase 4-C] per-layer logistic probe of CORRECTNESS
  plotting.py             # [Phase 2+] shared plot helpers (always draws baselines + control)
notebooks/
  00_smoke_test.ipynb     # load model, render one item, run image mode, report VRAM  <-- run this first
  01_behavioral_sweep.ipynb   # [Phase 2] accuracy vs compression, text baseline
  02_failure_sets.ipynb       # [Phase 3]
  03_interp.ipynb             # [Phase 4]
data/                     # gitignored: CSVs + .npz activations (point DATA_DIR at Drive on Colab to persist)
```

## Setup / how to run the smoke test
On Kaggle (Accelerator = **GPU T4 ×2**) or Colab (Runtime → GPU):
1. Upload/clone this repo so `config.py` sits at the repo root.
2. Open `notebooks/00_smoke_test.ipynb`, run top-to-bottom.
3. It installs deps (except torch — already on the runtime), reports GPU/VRAM,
   renders one BoolQ item, loads Qwen2.5-VL-7B, and prints the image-mode answer.
4. **Report back the per-GPU PEAK VRAM line** — that's the go/no-go for 2×T4.

Locally you can only lint/import the code (no CUDA GPU → cannot load the 7B model).

## Config knobs you'll actually touch
- `MODEL_ID` — swap Qwen2.5-VL-7B → Gemma-3 later (loader auto-detects the class).
- `LOAD_IN_4BIT=1` (env) — single-T4 fallback (see VRAM notes below).
- `DATASET_NAME` + `DATASET_FIELDS` + `normalize_answer` — swap BoolQ → an
  extractive set (e.g. DROP); only `src/inference.py:load_items` may need edits.
- `RENDER` / `FONT_SWEEP` — page box, ppi, font, and the Phase-2 size sweep.

## VRAM reality check (you asked me to flag compute assumptions)
- Qwen2.5-VL-7B in bf16 ≈ **~15–16 GB weights** + vision encoder + KV cache + image tokens.
- **Kaggle 2×T4 = 32 GB total.** `device_map="auto"` shards the model across both
  cards → comfortable. This is the recommended tier. ✅
- **Colab free single T4 = 16 GB.** 7B bf16 weights alone nearly fill it; adding
  vision tokens + KV cache will very likely **OOM**. ⚠️ On single-T4 either set
  `MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct` or `LOAD_IN_4BIT=1` (bitsandbytes).
- Activation capture keeps only `[num_layers+1, hidden]` per item (the answer
  position), ~0.3 MB/item — safe to store thousands of items on disk.

## Rigor requirements (enforced from Phase 2 on)
Every plot draws the chance baseline **and** the control-bucket curve; every
experiment prints a plain-English verdict and saves a per-layer CSV; before
trusting a result, check: above chance? control ≠ failure bucket? could
token-count / position / input-type explain it trivially?

## Reproducibility
Fixed `SEED`, greedy decoding (`do_sample=False`). Same items across modes.
