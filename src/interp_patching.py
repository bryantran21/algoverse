"""Position-restricted activation patching for causal localization.

Cache the residual stream from a **clean** (large-font) run and splice it into a
**corrupt** (small-font) run one layer at a time, measuring how much clean
behaviour is recovered.

Two design constraints, both load-bearing
-----------------------------------------
**1. Patch a restricted token span, never all positions.** Replacing a layer's
full output means every downstream layer computes on clean values, so recovery
is 1.000 at every layer including layer 0. That is a tautology, not a result.
`patch_layer` therefore always takes a position mask. `validate_design` asserts
the outcome is non-degenerate before a long run.

**2. Clean and corrupt inputs must align.** Qwen2.5-VL uses dynamic resolution,
so vision token counts key off image dimensions. With a fixed Typst page box
both font sizes render to the same canvas and produce identical token counts --
but this must be *verified*, and the two renders must also actually differ.
`check_alignment` does both.

Recovery is normalized so 0 = corrupt behaviour and 1 = clean behaviour.
"""
from __future__ import annotations

import numpy as np
import torch


def get_layers(model):
    """Decoder blocks, tolerating either wrapper depth."""
    if hasattr(model, "language_model"):
        return model.language_model.layers
    return model.model.language_model.layers


@torch.no_grad()
def logit_diff(model, inputs, yes_id: int, no_id: int) -> float:
    """logit(Yes) - logit(No) at the answer position, unpatched."""
    lg = model(**inputs, use_cache=False).logits[0, -1].float()
    return (lg[yes_id] - lg[no_id]).item()


@torch.no_grad()
def cache_residuals(model, inputs) -> dict[int, torch.Tensor]:
    """Residual stream at every layer for a single forward pass."""
    layers = get_layers(model)
    store, hooks = {}, []
    for i, layer in enumerate(layers):
        def mk(idx):
            def hook(mod, inp, out):
                store[idx] = (out[0] if isinstance(out, tuple) else out).detach().clone()
            return hook
        hooks.append(layer.register_forward_hook(mk(i)))
    try:
        model(**inputs, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return store


@torch.no_grad()
def patch_layer(model, inputs, store, layer_idx: int, pos_mask, yes_id: int, no_id: int) -> float:
    """Run `inputs` with layer `layer_idx` replaced by cached values, but ONLY at
    positions where `pos_mask` is True. Returns the resulting logit difference.
    """
    layers = get_layers(model)

    def hook(mod, inp, out):
        tup = isinstance(out, tuple)
        h = (out[0] if tup else out).clone()
        h[:, pos_mask, :] = store[layer_idx][:, pos_mask, :]
        return ((h,) + out[1:]) if tup else h

    hk = layers[layer_idx].register_forward_hook(hook)
    try:
        lg = model(**inputs, use_cache=False).logits[0, -1].float()
    finally:
        hk.remove()
    return (lg[yes_id] - lg[no_id]).item()


def masks_for(input_ids: torch.Tensor, vis_id: int) -> dict[str, np.ndarray]:
    """Standard position masks: the vision-token span, and the answer position."""
    ids = input_ids[0] if input_ids.ndim == 2 else input_ids
    n = ids.shape[0]
    ans = np.zeros(n, dtype=bool)
    ans[-1] = True
    return {"vision": (ids == vis_id).cpu().numpy(), "answer": ans}


def check_alignment(clean_inputs, corrupt_inputs, vis_id: int) -> dict:
    """Verify clean/corrupt inputs are patchable. Raises AssertionError if not."""
    lc = clean_inputs["input_ids"].shape[1]
    lx = corrupt_inputs["input_ids"].shape[1]
    vc = int((clean_inputs["input_ids"] == vis_id).sum())
    vx = int((corrupt_inputs["input_ids"] == vis_id).sum())
    assert lc == lx, f"sequence length mismatch: {lc} vs {lx}"
    assert vc == vx, f"vision token count mismatch: {vc} vs {vx}"
    return {"seq_len": lc, "n_vision": vc}


def validate_design(model, clean_inputs, corrupt_inputs, vis_id, yes_id, no_id,
                    probe_layers=(0, 14, 27), verbose=True) -> dict:
    """Probe a few layers under each mask and assert the result is not degenerate.

    Run this before any long patching job. If every masked patch still recovers
    1.0, the mask is not restricting anything and the run would be worthless.
    """
    check_alignment(clean_inputs, corrupt_inputs, vis_id)
    d_c = logit_diff(model, clean_inputs, yes_id, no_id)
    d_x = logit_diff(model, corrupt_inputs, yes_id, no_id)
    assert abs(d_c - d_x) > 1e-6, "clean and corrupt behave identically; nothing to recover"

    store = cache_residuals(model, clean_inputs)
    masks = masks_for(corrupt_inputs["input_ids"], vis_id)
    probe = {}
    for name, mask in masks.items():
        probe[name] = [
            (patch_layer(model, corrupt_inputs, store, L, mask, yes_id, no_id) - d_x) / (d_c - d_x)
            for L in probe_layers
        ]
        if verbose:
            vals = "  ".join(f"L{L} {v:+.3f}" for L, v in zip(probe_layers, probe[name]))
            print(f"{name.upper():7s}: {vals}", flush=True)
    del store
    torch.cuda.empty_cache()

    flat = all(abs(v - 1.0) < 1e-3 for vs in probe.values() for v in vs)
    assert not flat, "DEGENERATE: masked patching recovers 1.0 everywhere -- design is wrong"
    return probe


def patch_item(model, clean_inputs, corrupt_inputs, vis_id, yes_id, no_id,
               mask_names=("vision", "answer")) -> dict | None:
    """Full layer sweep for one item. Returns {mask: [recovery per layer]} or None
    if the item is unusable (misaligned, or clean == corrupt).
    """
    try:
        check_alignment(clean_inputs, corrupt_inputs, vis_id)
    except AssertionError:
        return None
    d_c = logit_diff(model, clean_inputs, yes_id, no_id)
    d_x = logit_diff(model, corrupt_inputs, yes_id, no_id)
    if abs(d_c - d_x) < 1e-6:
        return None

    n_layers = len(get_layers(model))
    masks = masks_for(corrupt_inputs["input_ids"], vis_id)
    store = cache_residuals(model, clean_inputs)
    out = {
        m: [(patch_layer(model, corrupt_inputs, store, L, masks[m], yes_id, no_id) - d_x)
            / (d_c - d_x) for L in range(n_layers)]
        for m in mask_names
    }
    del store
    torch.cuda.empty_cache()
    out["_meta"] = {"d_clean": d_c, "d_corrupt": d_x}
    return out


def crossover(vision_curve, answer_curve, thresh: float = 0.5) -> tuple[int | None, int | None]:
    """Layers where vision recovery falls below `thresh` and answer rises above it.

    The window between them is where information hands off from the visual
    representation to the decision. For Qwen2.5-VL at 14pt->5pt this is ~18-20.
    """
    v, a = np.asarray(vision_curve), np.asarray(answer_curve)
    v_drop = np.where(v < thresh)[0]
    a_rise = np.where(a > thresh)[0]
    return (int(v_drop[0]) if len(v_drop) else None,
            int(a_rise[0]) if len(a_rise) else None)
