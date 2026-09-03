# Transcription example renders (Appendix D)

The 5pt renders the model was asked to transcribe, for the three lowest-scoring
items in the corrected transcription distribution (n=62 failure set). Same
pipeline as the other renders (DejaVu Sans, 160×120 mm page, 8 mm margin, 150 DPI,
945×709 px). Pair each image with its gold text and the model's transcription;
the corrected scores come from `results/logit_lens/transcription_scores.npz`.

| image | item_id | gold label | corrected score | near-miss |
|---|---|---|---|---|
| `transcription_ex_item471_5pt.png` | 471 | yes | 0.941 (min) | echoes `Passage.`/`Question.` labels; normalizes `` ``R'' `` → `'R'` (×5) |
| `transcription_ex_item162_5pt.png` | 162 | yes | 0.946 | label echo; one glyph `≈` → `=` |
| `transcription_ex_item431_5pt.png` | 431 | yes | 0.950 | label echo only; content verbatim |

Every deduction is a formatting near-miss (label prefixes, quote/glyph
normalization), never a content misread — the model reads the render accurately
and still answers the question wrong.
