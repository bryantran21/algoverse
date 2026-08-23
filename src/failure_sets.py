"""Bucket items into failure / control sets from paired text/image inference.

The two sets are the backbone of every downstream experiment:

- **failure**: text-correct AND image-wrong. The text-correct filter is what
  attributes the failure to rendering rather than to question difficulty --
  the model demonstrably *can* answer the question when reading is easy.
- **control**: correct in BOTH modes. Without this set a flat image-mode
  logit-lens curve is uninterpretable, because the failure set is *selected*
  on image-wrong and a low final-layer value is partly guaranteed.

Also produces class-balanced variants. Note that at 5pt the failure set is
239 yes / 14 no, so `balance()` yields only 28 items -- too few for probing.
That imbalance is itself a finding (README finding 4), not a nuisance.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def bucket(items: list[dict], txt: dict, img: dict) -> dict[str, list[dict]]:
    """Split `items` by (text correct, image correct).

    Args:
        items: item dicts with 'item_id' and 'gold'.
        txt:   {item_id: result} from run_inference(..., "text").
        img:   {item_id: result} from run_inference(..., "image").

    Returns dict with keys: fail_all, ctrl_all, img_only, wrong_both.
    """
    by_id = {it["item_id"]: it for it in items}
    out = {"fail_all": [], "ctrl_all": [], "img_only": [], "wrong_both": []}
    for i in txt:
        if i not in img:
            continue
        t, m = txt[i]["correct"] == 1, img[i]["correct"] == 1
        key = ("ctrl_all" if m else "fail_all") if t else ("img_only" if m else "wrong_both")
        out[key].append(by_id[i])
    return out


def balance(subset: list[dict], seed: int = 0) -> list[dict]:
    """Downsample to equal yes/no counts. Returns 2*min(n_yes, n_no) items."""
    rng = np.random.default_rng(seed)
    pos = [it for it in subset if it["gold"] == "yes"]
    neg = [it for it in subset if it["gold"] == "no"]
    k = min(len(pos), len(neg))
    if k == 0:
        return []
    take = lambda xs: [xs[i] for i in rng.choice(len(xs), k, replace=False)]
    return take(pos) + take(neg)


def class_counts(subset: list[dict]) -> tuple[int, int]:
    """(n_yes, n_no) for a bucket."""
    yes = sum(1 for it in subset if it["gold"] == "yes")
    return yes, len(subset) - yes


def per_class_accuracy(items: list[dict], res: dict) -> dict[str, float]:
    """Accuracy split by gold label -- the comparison that reveals the bias.

    Information loss predicts both classes degrade; a directional bias predicts
    one degrades while the other may improve.
    """
    by_id = {it["item_id"]: it for it in items}
    out = {}
    for gold in ("yes", "no"):
        ids = [i for i in res if by_id.get(i, {}).get("gold") == gold]
        out[gold] = float(np.mean([res[i]["correct"] for i in ids])) if ids else float("nan")
    out["overall"] = float(np.mean([res[i]["correct"] for i in res]))
    return out


def false_no_rate(items: list[dict], img_res: dict) -> float:
    """Fraction of image-mode errors where gold was 'yes' (i.e. model said no)."""
    by_id = {it["item_id"]: it for it in items}
    errs = [i for i in img_res if img_res[i]["correct"] == 0]
    if not errs:
        return float("nan")
    return float(np.mean([by_id[i]["gold"] == "yes" for i in errs]))


def save(buckets: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(buckets, f)


def load(path: str | Path = "results/sets/sets_n2000.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def summarize(buckets: dict) -> None:
    for k, v in buckets.items():
        if not isinstance(v, list):
            continue
        y, n = class_counts(v)
        print(f"{k:12s} n={len(v):5d}  yes={y:5d}  no={n:5d}")
