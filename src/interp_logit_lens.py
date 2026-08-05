import numpy as np, torch, matplotlib.pyplot as plt
from src.activations import capture_activations

ANS = {"yes": 9454, "no": 2753}


def _lens_fns(model):
    lm_head    = model.get_output_embeddings()
    head_dtype = next(lm_head.parameters()).dtype
    final_norm = dict(model.named_modules())["model.language_model.norm"]
    norm_dev   = next(final_norm.parameters()).device
    head_dev   = next(lm_head.parameters()).device

    @torch.no_grad()
    def layerwise_correct_prob(item, mode, processor):
        rec = capture_activations(item, mode, model, processor, save=False)
        H = torch.tensor(rec["hidden"])
        gold_id = ANS[item["gold"]]
        probs = []
        for layer_vec in H:
            v = layer_vec.to(norm_dev, dtype=head_dtype)
            logits = lm_head(final_norm(v).to(head_dev))
            probs.append(torch.softmax(logits.float(), dim=-1)[gold_id].item())
        return np.array(probs)

    return layerwise_correct_prob


def run_logit_lens(model, processor, fail, ctrl, out_npz="results/logit_lens/logit_lens_curves.npz"):
    lcp = _lens_fns(model)
    n_ctrl = min(len(ctrl), len(fail))
    txt_curves, img_curves, ctrl_curves = [], [], []
    for it in fail:
        txt_curves.append(lcp(it, "text", processor))
        img_curves.append(lcp(it, "image", processor))
        torch.cuda.empty_cache()
    for it in ctrl[:n_ctrl]:
        ctrl_curves.append(lcp(it, "image", processor))
        torch.cuda.empty_cache()
    txt_curves, img_curves, ctrl_curves = map(np.array, (txt_curves, img_curves, ctrl_curves))
    if out_npz:
        np.savez(out_npz, txt=txt_curves, img=img_curves, ctrl_img=ctrl_curves)
    return txt_curves, img_curves, ctrl_curves


def boot_ci(curves, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(curves), size=(n_boot, len(curves)))
    means = curves[idx].mean(axis=1)
    return curves.mean(axis=0), np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def emergence(c, thresh=0.5):
    h = np.where(c > thresh)[0]
    return int(h[0]) if len(h) else None


def plot_logit_lens(txt_curves, img_curves, ctrl_curves, out_png="results/logit_lens/logit_lens.png"):
    txt_mean,  txt_lo,  txt_hi  = boot_ci(txt_curves)
    img_mean,  img_lo,  img_hi  = boot_ci(img_curves)
    ctrl_mean, ctrl_lo, ctrl_hi = boot_ci(ctrl_curves)
    layers = np.arange(len(txt_mean))
    plt.figure(figsize=(9, 5))
    for m, lo, hi, c, mk, lbl in [
        (txt_mean,  txt_lo,  txt_hi,  "green",     "o-", f"text, failure set (n={len(txt_curves)})"),
        (img_mean,  img_lo,  img_hi,  "crimson",   "s-", f"image, failure set (n={len(img_curves)})"),
        (ctrl_mean, ctrl_lo, ctrl_hi, "steelblue", "^-", f"image, control set (n={len(ctrl_curves)})"),
    ]:
        plt.plot(layers, m, mk, color=c, label=lbl, ms=4)
        plt.fill_between(layers, lo, hi, color=c, alpha=0.15)
    plt.axhline(0.5, color="gray", ls=":")
    plt.xlabel("layer (0 = embedding → final)")
    plt.ylabel("P(correct answer) via logit lens")
    plt.title("Where the answer emerges — Qwen2.5-VL-7B, BoolQ, 5pt")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=120)
    plt.show()
    for name, m in [("text/fail", txt_mean), ("image/fail", img_mean), ("image/ctrl", ctrl_mean)]:
        print(f"{name:12s} emerges at layer: {str(emergence(m)):5s} peak {m.max():.2f}")
