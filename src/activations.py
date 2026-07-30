"""Capture per-layer hidden states at the answer position.

Storage discipline: we keep ONLY the hidden state at a single position (the last
prompt token = where the first answer token is predicted) for every layer. That is
[num_layers+1, hidden] floats per item+mode — a few hundred KB — instead of the
full [seq, hidden] per layer, which would blow up disk fast.

answer position note: in text/combined mode the last prompt tokens are the natural
-language instruction; in image mode they are the same trailing instruction text,
so the last-token position is comparable across modes for Phase 1 infra. Phase 4
(patching) refines "answer position" with an explicit generation-position index.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import config
from src.inference import _messages_for, _flatten_images


def _act_path(item_id, mode) -> Path:
    return config.ACTS_DIR / f"item{item_id}_{mode}.npz"


@torch.no_grad()
def capture_activations(item: dict, mode: str, model, processor, save: bool = True):
    """Forward `item` in `mode`, return {'hidden': [L+1, H] float32, 'seq_len': int,
    'answer_pos': int}. Saves an .npz keyed by item_id+mode when save=True.

    hidden[layer] is the residual-stream state at answer_pos for that layer;
    layer 0 is the embedding output, layer L the final block.
    """
    msgs, images = _messages_for(item, mode)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs = _flatten_images([images])
    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    out = model(**inputs, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states                       # tuple len L+1, each [1, seq, H]
    answer_pos = hs[0].shape[1] - 1              # last (right-padded batch of 1)

    # Stack the single answer position across layers -> [L+1, H], to CPU float32.
    hidden = torch.stack([h[0, answer_pos, :] for h in hs], dim=0)
    hidden = hidden.to(torch.float32).cpu().numpy()

    rec = {
        "hidden": hidden,
        "seq_len": int(hs[0].shape[1]),
        "answer_pos": int(answer_pos),
        "item_id": int(item["item_id"]),
        "mode": mode,
    }
    if save:
        np.savez_compressed(_act_path(item["item_id"], mode), **rec)
    return rec


def load_activations(item_id, mode) -> dict:
    """Load a saved record; returns the same dict shape as capture_activations."""
    with np.load(_act_path(item_id, mode), allow_pickle=True) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def stack_bucket(item_ids, mode) -> np.ndarray:
    """Load many items -> [N, L+1, H] for probing / logit lens over a bucket."""
    return np.stack([load_activations(i, mode)["hidden"] for i in item_ids], axis=0)
