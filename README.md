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
readout), and no amount of black-box evaluation separates them.

**Headline result: in Qwen2.5-VL, it is readout failure, and specifically a
directional bias.** The rendered text is legible, the model transcribes it
near-perfectly, and yet rendering pushes the decision toward answering "no."
Adding a single constant to the Yes logit at inference recovers 21 accuracy
points. Causal patching localizes the corruption handoff to layers 18–20.

A second model (LLaVA-1.5-7B) shows the same *behavioral* gap for an entirely
different *mechanistic* reason — it cannot resolve the rendered text at all. Two
failure modes, indistinguishable from the outside. That contrast is the argument
for opening the model.

Everything runs on free GPU tiers (Kaggle 2×T4). Inference and hooks only. No
paid GPU, no training.

---

## Findings

Qwen2.5-VL-7B-Instruct, BoolQ, N=2000, passages rendered at 5pt unless noted.
Derived sets: **failure** n=253 (text-correct, image-wrong), **control** n=1450
(correct in both).

**1. The modality gap replicates.** Text mode 0.86, image mode 0.70.

**2. The answer surfaces only at the final layer, or never.** Logit lens on the
failure set: the correct answer emerges at layer 27 of 28 in text mode (peak
P=0.785) and never crosses 0.5 in image mode (peak 0.203). Image-mode *control*
items emerge at layer 27 as well (peak 0.737) — so late emergence is a property
of the model, not of the text modality. Layers are 0-indexed over 28 decoder
blocks. Readout is hook-based and validated to exact agreement with the model's
own final logits.

**3. It is not a perception problem (for Qwen).** Asked to transcribe the same
5pt images it answers incorrectly, the model achieves median 0.96 character
accuracy, with 79% of items above 0.90 and only 6% below 0.50. Inspection of the
low scorers shows most are span mismatch against the gold string rather than
genuine misreads, so true legibility is higher than the mean suggests. The text
is legible. The model reads it and then does not use it.

**4. Rendering induces a directional bias.** Under rendering, accuracy on
gold-yes items drops 17 points while gold-no items *gain* 7. 89% of all
image-mode errors are false "no" (χ²=129.7, p≈5e-30). Measuring the final-layer
yes−no logit difference directly: control items shift −1.33 (paired t,
p=1.7e-10, n=300), failure items shift −3.83 (p=3.5e-43, n=253) — roughly 3×
larger, and negative for 94–95% of items. On failure items the sign flips
(+1.31 → −1.81): the model does not become uncertain, it becomes confidently
wrong.

**5. The bias persists at legible font sizes.** Across 5/8/11pt the behavioral
trend is monotonic (gold-yes 0.70→0.82, gold-no 0.95→0.90, false-no rate
0.90→0.76). The logit shift is significant at every size — all three 95% CIs
exclude zero — including at 11pt, where the text is comfortably legible. Shift
magnitude itself is *not* monotonic (−0.96, −1.03, −0.62); 5pt and 8pt are
statistically indistinguishable.

**6. Causal localization: the handoff is at layers 18–20.** Position-restricted
activation patching, clean 14pt → corrupt 5pt at byte-identical canvas
dimensions (token counts verified equal), n=28 failure items. Patching the
**vision-token span** recovers ~0.95 of clean behaviour through layer 17, then
collapses — 0.43 at L18, 0.21 at L19, ~0 from L20 onward. Patching the **answer
position** is the mirror image: ~0 through L17, rising to 0.38 at L19, 0.75 at
L20, and 1.0 by L27. Information moves out of the vision tokens and into the
answer position across layers 18–20; repairing the visual representation after
that window does nothing. This explains why the answer only becomes decodable at
layer 27 — the decision is not formed at the answer position until after the
handoff.

**7. The bias is correctable with a constant.** Adding a fixed offset to the Yes
logit at inference, evaluated on the failure+control union (n=553, deliberately
failure-enriched, so the uncorrected baseline is not the model's true image-mode
accuracy). At the **independently measured** offset of 1.33 — the control-set
shift from finding 4, not a tuned value — accuracy goes **0.542 → 0.754**, a gain
of 21 points. Gold-yes recovers 0.423 → 0.720 while gold-no falls only
0.899 → 0.856; the correction fixes a bias rather than trading one class for the
other. A tuned optimum of 4.05 reaches 0.886 but is fit on the evaluation set and
is reported only for reference.

**8. A second model fails the same way behaviorally, for the opposite reason.**
LLaVA-1.5-7B drops 0.844 → 0.608 at 5pt, a 24-point gap that looks like a
replication. It is not. Transcription median is **0.039** at 5pt, and rises no
higher than 0.23 at 8/11/14pt — LLaVA's fixed 336×336 CLIP encoder means no font
size makes the render legible. Its accuracy is flat across font sizes, gold-yes
tracks its text-mode value almost exactly, and gold-no collapses to ~0.27: the
signature of a model defaulting to "yes" while ignoring the image. Qwen fails by
*readout*, LLaVA by *perception*, and the two are indistinguishable from
accuracy alone.

**9. Negative result.** Linear probing was planned and abandoned. The failure set
is 239 yes / 14 no, leaving too few negative-class items for a meaningful test —
a majority-class guesser scores 0.95, and the best AUROC obtained rested on three
items. The imbalance turned out to be the more informative finding (see 4).

Figures and raw arrays are under `results/`.

---

## Method notes

**Paired design.** Every item runs twice — once as text tokens, once as a Typst
render — with everything else held constant. The **text-correct filter** on the
failure set matters: it establishes the model *can* answer the question when
reading is easy, so image failure is attributable to rendering rather than to
question difficulty. The **control set** is what makes failure-set curves
interpretable — without it, a flat image-mode logit-lens curve says nothing,
since the set is *selected* on image-wrong.

**Full precision, not 4-bit.** Quantization perturbs activations, which is fatal
for interpretability. `device_map="auto"` in bf16 across 2×T4 instead.

**Logit lens.** Residual states are taken via forward hooks during the live pass
and projected through the final norm and unembedding. An earlier version
reconstructed from float32-stored hidden states and disagreed with the model's
own logits by up to 1.5 units; the hook-based version matches to five decimals.
Use `results/logit_lens_v2/`.

**Patching.** Patch a single layer at a *restricted token span*, never all
positions. An unrestricted version recovers 1.000 at every layer including layer
0 — degenerate by construction, since replacing layer 0's full output means every
subsequent layer computes on clean values. Verify before trusting any patching
run: clean and corrupt inputs must have identical `input_ids` length and
identical `<|image_pad|>` counts, and the two renders must differ (check pixel
diff, not just font setting).

**Transcription as a diagnostic.** Asking the model to transcribe the image it
just answered wrong separates readout failure from a perception floor, cheaply
and without touching internals. This is what caught the LLaVA result: without it,
LLaVA's symmetric degradation reads as evidence against the bias hypothesis
rather than as a model that cannot see.

---

## Layout

```
config.py                 # MODEL_ID, dataset, render defaults, paths, seed
src/
  rendering.py            # text QA -> PNG via Typst (fixed page box; font size = compression)
  inference.py            # load model; load_items; run text/image/combined modes
  activations.py          # answer-position hidden state per layer (superseded by hooks)
  failure_sets.py         # bucket items into failure / control
  interp_logit_lens.py    # per-layer P(correct) + bootstrap CIs
  interp_patching.py      # position-restricted causal patching
  interp_probe.py         # per-layer logistic probe (see finding 9)
  plotting.py             # shared plot helpers
notebooks/
  00_smoke_test.ipynb         # run this first
  01_behavioral_sweep.ipynb
  02_failure_sets.ipynb
  03_interp.ipynb             # logit lens, transcription, bias, font sweep
  04_mitigation.ipynb         # constant-offset correction
  05_patching_replication.ipynb
  06_replication.ipynb        # LLaVA-1.5-7B
  07_logit_lens_v2.ipynb      # hook-based, validated readout
results/
  logit_lens/             # v1 curves (superseded)
  logit_lens_v2/          # validated curves + figure
  sets/                   # n=2000 failure/control sets, bias stats, logit shifts
  dpi/                    # Qwen font sweep (5/8/11pt) + figure
  patching/               # position-restricted patch curves + figure
  mitigation/             # per-item logits, offset sweep + figure
  replication/            # LLaVA logits, font sweep
data/                     # gitignored: CSVs + .npz activations
```

---

## Running it

On Kaggle with **Accelerator = GPU T4 ×2** (check this — a batch job silently
inherits `None` and runs ~1200 s/item on CPU):

```
!git clone https://github.com/bryantran21/algoverse.git /kaggle/working/algoverse
```

Then `notebooks/00_smoke_test.ipynb` top to bottom.

**Session gotchas.** Kaggle wipes `/kaggle/working` on restart — re-clone and
reload every session, and `git config user.email/user.name` are wiped too. Long
runs (>20 min unattended) need Save Version → Save & Run All; interactive
sessions die on browser idle. Checkpoint inside long loops — a three-hour cell
that only writes at the end will lose everything. Assert `torch.cuda.is_available()`
at the top of every notebook.

**Model specifics.** Final norm at `model.language_model.norm`; decoder blocks at
`model.language_model.layers`. Qwen answer token ids are Yes=9454, No=2753;
LLaVA-1.5 uses Yes=3869, No=1939 and a `USER: <image>\n... ASSISTANT:` prompt
format. With `device_map="auto"`, `lm_head` and the final norm may sit on
different devices than `model.device` reports — move tensors to
`next(module.parameters()).device`.

Locally you can only lint and import (no CUDA → cannot load a 7B model).

---

## Config knobs

- `MODEL_ID` — swap the VLM (loader auto-detects the class). Note that answer
  token ids and prompt format are model-specific and must be set by hand.
- `DATASET_NAME` + `DATASET_FIELDS` + `normalize_answer` — swap BoolQ for another
  binary QA set. The pipeline assumes a yes/no answer space; an extractive
  dataset needs more than a loader change.
- `RENDER` / `FONT_SWEEP` — page box, ppi, font, and sweep points.
- `BATCH_SIZE` — 2 fits at 5pt on 2×T4. Larger fonts produce more vision tokens
  under dynamic resolution, so they are slower and tighter on memory, not faster.

---

## Rigor requirements

Every plot draws the chance baseline and the control-bucket curve. Every
experiment saves **per-item arrays**, not just summary statistics — bootstrap CIs
are impossible to add later otherwise. Save under distinct filenames per run;
reusing variable names across a control run and a failure run once overwrote the
control arrays with failure data, which was only caught by an implausible
downstream number.

Before trusting a result: is it above chance? Does control differ from failure?
Could token count, position, or input type explain it trivially? Is the effect
partly guaranteed by how the set was selected? For any patching or intervention,
is the "effect" achievable by construction?

**Reproducibility.** Fixed `SEED`, greedy decoding (`do_sample=False`), same items
across modes. The 62/369 failure/control split at N=500 reproduced exactly across
three separate sessions and across a transformers image-processor change. Headline
results use N=2000 (253/1450).

---

## Limitations

- **Single dataset.** BoolQ's yes-skew is a live confound for the directional
  bias; whether it holds on a class-balanced binary QA set is untested.
- **Bias generalization is open.** LLaVA-1.5 cannot resolve the renders at any
  tested font size, so it tests the perception floor rather than the bias.
  Confirming or refuting generalization needs another high- or dynamic-resolution
  VLM.
- **Failure-set selection effect.** The set is selected on image-wrong, so a low
  final-layer logit-lens value there is partly guaranteed. The control comparison
  is what makes it interpretable; the layer index is the informative part, not
  the magnitude.
- **Patching n=28**, single clean/corrupt font pair (14pt→5pt). Layer resolution
  of the 18–20 handoff would tighten with more items and more font pairs.
- **Mitigation offset** was measured on the control set and applied to a
  failure-enriched union drawn from the same distribution; it is not validated
  out-of-distribution.
- **Shift magnitude is not monotonic** across font size.
