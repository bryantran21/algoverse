import numpy as np, torch, matplotlib.pyplot as plt
from src.activations import capture_activations

lm_head = model.get_output_embeddings()
head_dtype = next(lm_head.parameters()).dtype
# final_norm already found above; re-grab to be safe
final_norm = dict(model.named_modules())["model.language_model.norm"]
ANS = {"yes": 9454, "no": 2753}

@torch.no_grad()
def layerwise_correct_prob(item, mode):
    rec = capture_activations(item, mode, model, processor, save=False)
    H = torch.tensor(rec["hidden"])
    gold_id = ANS[item["gold"]]
    probs = []
    for layer_vec in H:
        v = layer_vec.to(model.device, dtype=head_dtype)
        logits = lm_head(final_norm(v))
        p = torch.softmax(logits.float(), dim=-1)[gold_id].item()
        probs.append(p)
    return np.array(probs)

txt_curves, img_curves = [], []
for it in fail:
    txt_curves.append(layerwise_correct_prob(it, "text"))
    img_curves.append(layerwise_correct_prob(it, "image"))
    torch.cuda.empty_cache()

txt_mean, img_mean = np.mean(txt_curves,axis=0), np.mean(img_curves,axis=0)
layers = np.arange(len(txt_mean))

plt.figure(figsize=(9,5))
plt.plot(layers, txt_mean, "o-", color="green",   label="text mode (model gets these right)")
plt.plot(layers, img_mean, "s-", color="crimson", label="image mode (model fails these)")
plt.axhline(0.5, color="gray", ls=":")
plt.xlabel("layer (0 = embedding → final)")
plt.ylabel("P(correct answer) via logit lens")
plt.title(f"Where the answer emerges — failure set (n={len(fail)})")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("/kaggle/working/logit_lens.png", dpi=120); plt.show()

def emergence(c): 
    h = np.where(c>0.5)[0]; return int(h[0]) if len(h) else None
print(f"text  emerges at layer: {emergence(txt_mean)}  (peak {txt_mean.max():.2f})")
print(f"image emerges at layer: {emergence(img_mean)}  (peak {img_mean.max():.2f})")
