"""Load the VLM and run text / image / combined inference.

Public API:
    load_vl_model()                     -> (model, processor)
    load_items(n=None)                  -> list[dict]  (swappable dataset loader)
    run_inference(items, mode, model, processor, out_csv=None) -> list[dict]

Each result dict: item_id, mode, prediction, correct, gold, input_tokens,
total_tokens, raw_text. Results append to out_csv after every batch so a
Colab/Kaggle timeout never loses completed work.
"""
from __future__ import annotations

import csv
import io
import random
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

import config
from src.rendering import render_qa_to_image

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_vl_model():
    """Load MODEL_ID with device_map='auto' (shards across 2x T4). Returns
    (model, processor). 4-bit path enabled via config.LOAD_IN_4BIT."""
    from transformers import AutoProcessor

    _seed_everything(config.SEED)
    dtype = _DTYPES[config.TORCH_DTYPE]

    load_kwargs = dict(torch_dtype=dtype, device_map=config.DEVICE_MAP)
    if config.LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs.pop("torch_dtype", None)

    model = _auto_model(config.MODEL_ID).from_pretrained(config.MODEL_ID, **load_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(config.MODEL_ID)
    return model, processor


def _auto_model(model_id: str):
    """Pick the right *ConditionalGeneration class for the model family."""
    mid = model_id.lower()
    if "qwen2.5-vl" in mid or "qwen2_5_vl" in mid:
        from transformers import Qwen2_5_VLForConditionalGeneration as M
    elif "qwen2-vl" in mid:
        from transformers import Qwen2VLForConditionalGeneration as M
    elif "gemma-3" in mid or "gemma3" in mid:
        from transformers import Gemma3ForConditionalGeneration as M
    else:
        from transformers import AutoModelForImageTextToText as M
    return M


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Dataset (swap the body of load_items when moving off BoolQ)
# --------------------------------------------------------------------------- #
def load_items(n: int | None = None) -> list[dict]:
    """Return a list of {item_id, passage, question, gold} dicts.

    gold is the canonical label from config.normalize_answer. To swap datasets:
    change config.DATASET_* and, if the schema differs, this function only.
    """
    from datasets import load_dataset

    f = config.DATASET_FIELDS
    ds = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    items = []
    for i, row in enumerate(ds):
        items.append(
            {
                "item_id": i,
                "passage": row[f["passage"]],
                "question": row[f["question"]],
                "gold": config.normalize_answer(row[f["answer"]]),
            }
        )
    return items


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
_INSTRUCTION = (
    "Read the passage and answer the question with a single word: "
    f"{' or '.join(config.ANSWER_LABELS)}."
)


def _messages_for(item: dict, mode: str) -> tuple[list, list]:
    """Build chat messages + the list of PIL images for one item and mode."""
    assert mode in {"text", "image", "combined"}, mode
    images: list[Image.Image] = []
    content: list[dict] = []

    if mode in {"image", "combined"}:
        png = render_qa_to_image(item["passage"], item["question"])
        images.append(Image.open(io.BytesIO(png)).convert("RGB"))
        content.append({"type": "image"})

    if mode == "image":
        text = f"{_INSTRUCTION} The passage and question are in the image."
    elif mode == "text":
        text = f"{_INSTRUCTION}\n\nPassage: {item['passage']}\n\nQuestion: {item['question']}"
    else:  # combined
        text = (
            f"{_INSTRUCTION} The passage and question are also shown in the image.\n\n"
            f"Passage: {item['passage']}\n\nQuestion: {item['question']}"
        )
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}], images


def _parse_prediction(text: str) -> str:
    """Map raw generated text to a canonical label (or '' if none matched)."""
    low = text.strip().lower()
    for label in config.ANSWER_LABELS:
        if low.startswith(label) or f" {label}" in f" {low}":
            return label
    return ""


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_inference(
    items: Iterable[dict],
    mode: str,
    model,
    processor,
    out_csv: str | Path | None = None,
    batch_size: int | None = None,
) -> list[dict]:
    """Run `mode` inference over items. Appends each batch to out_csv."""
    items = list(items)
    batch_size = batch_size or config.BATCH_SIZE
    results: list[dict] = []

    writer, fh = _open_csv(out_csv) if out_csv else (None, None)
    try:
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            rows = _run_batch(batch, mode, model, processor)
            results.extend(rows)
            if writer:
                for r in rows:
                    writer.writerow(r)
                fh.flush()
    finally:
        if fh:
            fh.close()
    return results


def _run_batch(batch, mode, model, processor) -> list[dict]:
    texts, all_images = [], []
    for item in batch:
        msgs, images = _messages_for(item, mode)
        texts.append(
            processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )
        all_images.append(images)

    # process_vision_info lives in qwen_vl_utils; fall back to raw image lists.
    image_inputs = _flatten_images(all_images)
    inputs = processor(
        text=texts,
        images=image_inputs if image_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]
    gen = model.generate(
        **inputs,
        max_new_tokens=config.MAX_NEW_TOKENS,
        do_sample=not config.GREEDY,
    )
    new_tokens = gen[:, input_len:]
    decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)

    rows = []
    for item, raw, seq, in_ids in zip(batch, decoded, gen, inputs["input_ids"]):
        pred = _parse_prediction(raw)
        rows.append(
            {
                "item_id": item["item_id"],
                "mode": mode,
                "prediction": pred,
                "gold": item["gold"],
                "correct": int(pred == item["gold"]),
                "input_tokens": int((in_ids != processor.tokenizer.pad_token_id).sum())
                if processor.tokenizer.pad_token_id is not None
                else int(in_ids.numel()),
                "total_tokens": int(seq.numel()),
                "raw_text": raw.strip().replace("\n", " ")[:200],
            }
        )
    return rows


def _flatten_images(all_images):
    out = []
    for imgs in all_images:
        out.extend(imgs)
    return out


_CSV_COLS = [
    "item_id", "mode", "prediction", "gold", "correct",
    "input_tokens", "total_tokens", "raw_text",
]


def _open_csv(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=_CSV_COLS)
    if not exists:
        writer.writeheader()
    return writer, fh
