# Example renders (appendix figure)

One BoolQ passage (validation `item_id` 1, gold = yes) rendered at 5 / 8 / 11 / 14 pt
with the pipeline's settings: DejaVu Sans, fixed 160×120 mm page, 8 mm margin, 150 DPI.

- **Canvas: 945×709 px at every font size** — the page box is fixed, so only text
  density changes, not image dimensions.
- **Qwen2.5-VL vision tokens: 850 at every size** — the token count is a function of
  image dimensions only, and those are identical across sizes (verified equal for
  5pt vs 14pt in the patching alignment check).
- Note the passage overflows the canvas by 14 pt: larger type fits *less* of the
  passage, which is part of why bigger fonts do not help LLaVA (see finding 8).

Regenerate with `scripts`-equivalent code in `src/rendering.py`
(`render_qa_to_image`, fixed page box from `config.RENDER`).
