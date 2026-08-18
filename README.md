# Visualized-text modality gap in VLMs — mechanistic analysis

When text is rendered as an image and shown to a vision-language model instead of
passed as tokens, accuracy drops sharply. Prior work establishes *that* this
happens. This repo asks *where inside the model* it happens, and *what kind* of
failure it is.

The two candidate explanations make different predictions inside the network:

- **Information loss** — the vision encoder never resolves the glyphs, so the
  answer is not present in the residual stream at any depth.
- **Readout failure** — the text is encoded fine, and the language model fails to
  use it when forming an answer.

These call for opposite interventions (better encoders vs. better cross-modal
readout), and no amount of black-box evaluation separates them. We separate them
with logit lens, a transcription control, and direct measurement of the
final-layer decision.

**Headline result: it is readout failure, and specifically a directional bias.**
Rendering does not push the model toward uncertainty — it pushes it toward
answering "no."

Everything runs on free GPU tiers (Kaggle 2×T4). Inference and a small amount of
linear algebra only. No paid GPU, no training.

---

## Findings

Qwen2.5-VL-7B-Instruct, BoolQ, N=2000, passages rendered at 5pt unless noted.

**1. The modality gap replicates.** Text mode 0.86, image mode 0.70.

**2. The answer surfaces only at the final layer, or never.** Logit lens on the
failure set: the correct answer emerges at layer 28 of 28 in text mode (peak
P=0.75) and never emerges in image mode (peak 0.19). Image-mode *control* items
emerge at layer 28 as well (peak 0.83) — so late emergence is a property of the
model, not of the text modality.

**3. It is not a perception problem.** Asked to transcribe the same 5pt images it
answers incorrectly, the model achieves median 0.96 character accuracy, with 79%
of items above 0.90. The text is legible. The model reads it and then does not
use it.

**4. Rendering induces a directional bias.** Under rendering, accuracy on
gold-yes items drops 17 points while gold-no items *gain* 7. 89% of all
image-mode errors are false "no" (χ²=129.7, p≈5e-30). Measuring the final-layer
yes−no logit difference directly: control items shift −0.87 (paired t, p=1.7e-10),
failure items shift −3.12 (p=3.5e-43) — 3.6× larger, and negative for 95% of
items. The bias concentrates exactly where the errors are.

**5. The bias persists at legible font sizes.** Across 5/8/11pt the behavioral
trend is monotonic (gold-yes 0.70→0.82, gold-no 0.95→0.90, false-no rate
0.90→0.76). The logit shift is significant at every size — all three 95% CIs
exclude zero — including at 11pt, where the text is comfortably legible. Shift
magnitude itself is *not* monotonic (−0.96, −1.03, −0.62); 5pt and 8pt are
statistically indistinguishable.

**Negative result.** Linear probing was planned and abandoned. The failure set is
239 yes / 14 no, leaving too few negative-class items for a meaningful test — a
majority-class guesser scores 0.95. The imbalance turned out to be the more
informative finding.

Figures and raw arrays are under `results/`.

---

## Method notes

**Paired design.** Every item runs twice — once as text tokens, once as a Typst
render — with everything else held constant. Two derived sets:

- **Failure set** (n=253): text-correct, image-incorrect. The text-correct filter
  matters: it establishes the model *can* answer the question when reading is
  easy, so image failure is attributable to rendering rather than to question
  difficulty.
- **Control set** (n=1450): correct in both modes. This is what makes failure-set
  curves interpretable — without it, a flat image-mode logit-lens curve is
  uninterpretable, since the set is *selected* on image-wrong.

**Full precision, not 4-bit.** Quantization perturbs activations, which is fatal
for interpretability. `device_map="auto"` in bf16 across 2×T4 instead.

**Activation capture** keeps only `[num_layers+1, hidden]` per item — the residual
stream at the final prompt position, where the first answer token is predicted.
~0.3 MB/item, so thousands of items fit on disk.

---

## Layout

```
config.py                 # MODEL_ID, dataset, render defaults, paths, seed
src/
  rendering.py            # text QA -> PNG via Typst (fixed page box; font size = compression)
  inference.py            # load model; load_items; run text/image/combined modes
  activations.py          # capture answer-position hidden state per layer
  failure_sets.py         # bucket items into failure / control
  interp_logit_lens.py    # project hidden states -> vocab; P(correct) vs layer; bootstrap CIs
  interp_patching.py      # [in progress] causal localization via activation patching
  interp_probe.py         # per-layer logistic probe (see negative result above)
  plotting.py             # shared plot helpers (always draws baselines + control)
notebooks/
  00_smoke_test.ipynb     # load model, render one item, report VRAM  <-- run this first
  01_behavioral_sweep.ipynb
  02_failure_sets.ipynb
  03_interp.ipynb
results/
  logit_lens/             # curves, transcription scores, probe scores
  sets/                   # n=2000 failure/control sets, bias stats, logit shifts
  dpi/                    # font sweep (5/8/11pt) + figure
data/                     # gitignored: CSVs + .npz activations
```

---

## Running it

On Kaggle with Accelerator = GPU T4 ×2:

```python
!git clone https://{TOKEN}@github.com/bryantran21/algoverse.git /kaggle/working/algoverse
```

Then `notebooks/00_smoke_test.ipynb` top to bottom. It installs deps (torch is
already on the runtime), reports VRAM, renders one BoolQ item, loads the model,
and prints the image-mode answer.

**Session gotchas.** Kaggle wipes `/kaggle/working` on restart — re-clone and
reload every session, and `git config user.email/user.name` are wiped too. Long
runs (>20 min unattended) need Save Version → Save & Run All, since interactive
sessions die on browser idle. Checkpoint inside long loops; a two-hour cell that
only writes at the end will lose everything.

**Model specifics.** The final norm lives at `model.language_model.norm`. Answer
token ids are Yes=9454, No=2753. With `device_map="auto"`, `lm_head` and the final
norm may sit on different devices than `model.device` reports — move tensors to
`next(module.parameters()).device`, not to `model.device`.

Locally you can only lint and import (no CUDA → cannot load the 7B model).

---

## Config knobs

- `MODEL_ID` — swap Qwen2.5-VL-7B for another VLM (loader auto-detects the class).
- `DATASET_NAME` + `DATASET_FIELDS` + `normalize_answer` — swap BoolQ for another
  binary QA set. Note that the hardcoded Yes/No token ids are model-specific and
  the pipeline assumes a yes/no answer space; an extractive dataset needs more
  than a loader change.
- `RENDER` / `FONT_SWEEP` — page box, ppi, font, and the font-size sweep points.
- `BATCH_SIZE` — 2 fits at 5pt on 2×T4. Larger fonts produce more vision tokens
  (dynamic resolution), so they are slower and tighter on memory, not faster.

---

## Rigor requirements

Every plot draws the chance baseline and the control-bucket curve. Every
experiment saves per-layer arrays, not just summary statistics — bootstrap CIs
are impossible to add later otherwise. Before trusting a result: is it above
chance? Does control differ from failure? Could token count, position, or input
type explain it trivially? Is the effect partly guaranteed by how the set was
selected?

**Reproducibility.** Fixed `SEED`, greedy decoding (`do_sample=False`), same items
across modes. The 62/369 split at N=500 reproduced exactly across three separate
sessions and across a transformers image-processor change.

---

## Limitations

- Single model, single dataset. Whether the "no" bias is BoolQ-specific is open.
- The failure set is selected on image-wrong, so a low final-layer logit-lens
  value there is partly guaranteed by construction. The control comparison is
  what makes it interpretable; the layer index is the informative part, not the
  magnitude.
- Logit-shift magnitude is not monotonic across font size.
- Causal claims await activation patching. Everything here is correlational.
