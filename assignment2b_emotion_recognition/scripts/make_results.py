"""Tables and plots from the finished runs.

    python scripts/make_results.py --results Results

Writes to Results/:
  accuracy_table.md / .csv    accuracy and macro-F1 (mean +- std over seeds) for every variant and split
  per_class_recall.csv        recall of each emotion, main variants, mean over seeds
  separability.csv            silhouette score of the learned representations (cosine), per block and split
  noise_robustness.csv        word-split accuracy with white noise added to the test speech
  fusion_modality_reliance.csv  fusion accuracy with the speech or the text of each clip swapped for another's
  fusion_gate.csv             GMU: average share of the fused vector taken from speech, per emotion
  failure_cases.csv           misclassified test clips of the fusion model, most confident first
  attention_time_centre.csv   cross-modal attention: where in the clip each text token looks (0 start, 1 end)
  plots/                      confusion matrices, t-SNE of each block, training curves, noise curve, layer weights,
                              cross-modal attention maps
"""
import argparse
import csv
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.data import EMOTION_NAMES, EMOTIONS  # noqa: E402

SPLITS = ["word", "oaf2yaf", "yaf2oaf"]
SPLIT_NAMES = {"word": "unseen words", "oaf2yaf": "OAF -> YAF", "yaf2oaf": "YAF -> OAF"}
MAIN = ["speech", "text", "fusion"]
COLOURS = ["#d62728", "#8c564b", "#9467bd", "#ff7f0e", "#7f7f7f", "#2ca02c", "#1f77b4"]


def load_runs(results):
    runs = {}
    for p in glob.glob(os.path.join(results, "runs", "*", "*", "seed*", "metrics.json")):
        seed_dir = os.path.dirname(p)
        split_dir = os.path.dirname(seed_dir)
        variant = os.path.basename(os.path.dirname(split_dir))
        m = json.load(open(p))
        m["dir"] = seed_dir
        runs.setdefault((variant, os.path.basename(split_dir)), []).append(m)
    return runs


def ms(vals):
    v = np.array(vals, dtype=float)
    return f"{100 * v.mean():.1f} ± {100 * v.std():.1f}"


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def accuracy_tables(runs, out):
    variants = [v for v in ["speech", "speech_logmel", "speech_spknorm", "speech_mean", "speech_mean_aug",
                            "speech_logmel_aug", "text", "fusion", "fusion_concat", "fusion_xattn"]
                if any((v, s) in runs for s in SPLITS)]
    rows = [["variant"] + [f"{SPLIT_NAMES[s]} {k}" for s in SPLITS for k in ("accuracy", "macro_f1")]]
    md = ["| Variant | " + " | ".join(f"{SPLIT_NAMES[s]}: accuracy | macro-F1" for s in SPLITS) + " |",
          "|---" * (1 + 2 * len(SPLITS)) + "|"]
    for v in variants:
        cells = []
        for s in SPLITS:
            r = runs.get((v, s), [])
            cells += [ms([x["accuracy"] for x in r]) if r else "–", ms([x["macro_f1"] for x in r]) if r else "–"]
        rows.append([v] + cells)
        md.append(f"| {v} | " + " | ".join(cells) + " |")
    n = max(len(r) for r in runs.values())
    md.append(f"\nMean ± standard deviation over {n} seeds, in %. Chance level is 14.3 % (7 balanced classes).")
    write_csv(os.path.join(out, "accuracy_table.csv"), rows)
    open(os.path.join(out, "accuracy_table.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


def per_class(runs, out):
    rows = [["variant", "split"] + EMOTIONS]
    for v in MAIN:
        for s in SPLITS:
            r = runs.get((v, s))
            if r:
                rows.append([v, s] + [round(100 * np.mean([x["per_class_recall"][e] for x in r]), 1) for e in EMOTIONS])
    write_csv(os.path.join(out, "per_class_recall.csv"), rows)


def confusions(runs, out):
    for s in SPLITS:
        avail = [v for v in MAIN if (v, s) in runs]
        if not avail:
            continue
        fig, axes = plt.subplots(1, len(avail), figsize=(4.6 * len(avail), 4.4))
        for ax, v in zip(np.atleast_1d(axes), avail):
            cm = np.sum([np.array(x["confusion"]) for x in runs[(v, s)]], 0)
            cm = cm / cm.sum(1, keepdims=True).clip(min=1)
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            for i in range(len(EMOTIONS)):
                for j in range(len(EMOTIONS)):
                    ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > 0.5 else "black")
            ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, fontsize=8)
            ax.set_yticks(range(len(EMOTIONS)), EMOTIONS, fontsize=8)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            acc = np.mean([x["accuracy"] for x in runs[(v, s)]])
            ax.set_title(f"{v} ({100 * acc:.1f} %)", fontsize=10)
        fig.suptitle(f"Confusion matrices, row-normalised %, all seeds ({SPLIT_NAMES[s]})", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "plots", f"confusion_{s}.png"), dpi=140)
        plt.close(fig)


def separability(runs, out):
    """t-SNE of the block outputs (seed 0) and silhouette scores in the original space (all seeds)."""
    rows = [["split", "model", "block", "silhouette_mean", "silhouette_std"]]
    blocks = [("speech", "temporal", "speech: temporal block"), ("text", "contextual", "text: contextual block"),
              ("fusion", "fusion", "fusion block")]
    for s in SPLITS:
        panels = []
        for v, key, title in blocks:
            r = runs.get((v, s))
            if not r:
                continue
            sil = []
            for x in r:
                d = np.load(os.path.join(x["dir"], "reps.npz"))
                if len(set(d["labels"])) > 1:
                    sil.append(silhouette_score(d[key], d["labels"], metric="cosine"))
            rows.append([s, v, key, round(float(np.mean(sil)), 3), round(float(np.std(sil)), 3)])
            d = np.load(os.path.join(sorted(r, key=lambda x: x["seed"])[0]["dir"], "reps.npz"))
            panels.append((title, d[key], d["labels"], d["speakers"], float(np.mean(sil))))
        if not panels:
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.6))
        for ax, (title, x, y, spk, sil) in zip(np.atleast_1d(axes), panels):
            emb = TSNE(2, perplexity=min(30, len(x) // 4), init="pca", random_state=0).fit_transform(x)
            for i, e in enumerate(EMOTIONS):
                for sp, mk in (("OAF", "o"), ("YAF", "^")):
                    k = (y == i) & (spk == sp)
                    if k.any():
                        ax.scatter(emb[k, 0], emb[k, 1], s=9, c=COLOURS[i], marker=mk, alpha=0.75,
                                   label=EMOTION_NAMES[e] if sp == "OAF" or not (spk == "OAF").any() else None)
            ax.set_title(f"{title}\nsilhouette {sil:.2f}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        np.atleast_1d(axes)[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5), markerscale=1.5)
        fig.suptitle(f"t-SNE of test-set representations ({SPLIT_NAMES[s]}; circles OAF, triangles YAF)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "plots", f"tsne_{s}.png"), dpi=140)
        plt.close(fig)
    write_csv(os.path.join(out, "separability.csv"), rows)


def noise(runs, out):
    rows = [["variant", "snr_db", "accuracy_mean", "accuracy_std"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for v in ["speech", "fusion", "fusion_concat", "fusion_xattn"]:
        r = runs.get((v, "word"))
        if not r or not r[0].get("noise"):
            continue
        levels = ["clean"] + sorted(r[0]["noise"], key=lambda k: -int(k))
        means, stds = [], []
        for lv in levels:
            acc = [x["accuracy"] if lv == "clean" else x["noise"][lv]["accuracy"] for x in r]
            means.append(100 * np.mean(acc))
            stds.append(100 * np.std(acc))
            rows.append([v, lv, round(means[-1], 1), round(stds[-1], 1)])
        ax.errorbar(range(len(levels)), means, yerr=stds, marker="o", capsize=3, label=v)
        ax.set_xticks(range(len(levels)), [lv if lv == "clean" else f"{lv} dB" for lv in levels])
    if len(rows) > 1:
        ax.axhline(100 / 7, ls="--", c="grey", lw=0.8, label="chance")
        ax.set_ylabel("accuracy (%)")
        ax.set_title("Unseen-word test set with white noise added", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "plots", "noise_robustness.png"), dpi=140)
        write_csv(os.path.join(out, "noise_robustness.csv"), rows)
    plt.close(fig)


def reliance(runs, out):
    """Fusion accuracy when one modality is replaced by that of a random other test clip."""
    rows = [["variant", "split", "both_intact", "speech_shuffled", "text_shuffled"]]
    for v in ["fusion", "fusion_concat", "fusion_xattn"]:
        for s in SPLITS:
            r = [x for x in runs.get((v, s), []) if "modality_ablation" in x]
            if r:
                rows.append([v, s, ms([x["accuracy"] for x in r]),
                             ms([x["modality_ablation"]["speech_shuffled"] for x in r]),
                             ms([x["modality_ablation"]["text_shuffled"] for x in r])])
    write_csv(os.path.join(out, "fusion_modality_reliance.csv"), rows)


def gate(runs, out):
    rows = [["split", "mean"] + EMOTIONS]
    for s in SPLITS:
        r = [x for x in runs.get(("fusion", s), []) if "gate_speech_share" in x]
        if r:
            rows.append([s, round(np.mean([x["gate_speech_share"]["mean"] for x in r]), 3)]
                        + [round(np.mean([x["gate_speech_share"][e] for x in r]), 3) for e in EMOTIONS])
    write_csv(os.path.join(out, "fusion_gate.csv"), rows)


def curves(runs, out):
    fig, axes = plt.subplots(1, len(MAIN), figsize=(4.6 * len(MAIN), 3.4), sharey=True)
    for ax, v in zip(axes, MAIN):
        for s, c in zip(SPLITS, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            for x in runs.get((v, s), []):
                h = json.load(open(os.path.join(x["dir"], "history.json")))
                ep = [e["epoch"] for e in h]
                lab = SPLIT_NAMES[s] if x["seed"] == 0 else None
                ax.plot(ep, [100 * e["val_accuracy"] for e in h], c=c, alpha=0.8, label=lab)
        ax.axhline(100 / 7, ls="--", c="grey", lw=0.8)
        ax.set_title(v)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("validation accuracy (%)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "plots", "validation_curves.png"), dpi=140)
    plt.close(fig)


def layer_plot(runs, out):
    r = runs.get(("speech", "word"))
    if not r or not r[0].get("layer_weights"):
        return
    w = np.mean([list(x["layer_weights"].values())[0] for x in r], 0)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(range(len(w)), w, color="#1f77b4")
    ax.set_xlabel("WavLM layer (0 = CNN output)")
    ax.set_ylabel("learned weight")
    ax.set_title("Speech model: weight of each WavLM layer (unseen words, mean of seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "plots", "wavlm_layer_weights.png"), dpi=140)
    plt.close(fig)


def attention_plots(runs, out):
    """Cross-modal attention (fusion_xattn): mean time centre per token, and example maps (unseen words, seed 0)."""
    order = ["[CLS]", "say", "the", "word", "target word", "[SEP]"]
    rows = [["split"] + order]
    for s in SPLITS:
        r = [x for x in runs.get(("fusion_xattn", s), []) if "attention_time_centre" in x]
        if r:
            rows.append([s] + [round(float(np.mean([x["attention_time_centre"].get(k, np.nan) for x in r])), 3)
                               for k in order])
    if len(rows) == 1:
        return
    write_csv(os.path.join(out, "attention_time_centre.csv"), rows)
    r = sorted(runs.get(("fusion_xattn", "word"), []), key=lambda x: x["seed"])
    if not r or not os.path.exists(os.path.join(r[0]["dir"], "attention.npz")):
        return
    d = np.load(os.path.join(r[0]["dir"], "attention.npz"))
    toks = json.load(open(os.path.join(r[0]["dir"], "attention_tokens.json")))
    ids = list(d["ids"])[:6]
    fig, axes = plt.subplots(len(ids), 1, figsize=(8, 1.25 * len(ids) + 0.6))
    for ax, uid in zip(np.atleast_1d(axes), ids):
        m = d[f"map_{uid}"]
        ax.imshow(m, aspect="auto", cmap="magma", interpolation="nearest",
                  extent=[0, m.shape[1] * 0.02, m.shape[0] - 0.5, -0.5])
        ax.set_yticks(range(m.shape[0]), toks[uid], fontsize=7)
        ax.set_title(uid, fontsize=8, loc="left")
    np.atleast_1d(axes)[-1].set_xlabel("time in the clip (s)")
    fig.suptitle("Cross-modal attention: which speech frames each text token attends to", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "plots", "attention_maps.png"), dpi=140)
    plt.close(fig)


def failures(runs, out):
    rows = [["split", "seed", "id", "speaker", "word", "true", "pred", "confidence", "speech_model_pred"]]
    for s in SPLITS:
        for x in sorted(runs.get(("fusion", s), []), key=lambda x: x["seed"])[:1]:
            sp = {}
            r = [y for y in runs.get(("speech", s), []) if y["seed"] == x["seed"]]
            if r:
                sp = {row["id"]: row["pred"] for row in csv.DictReader(open(os.path.join(r[0]["dir"], "predictions.csv")))}
            wrong = [row for row in csv.DictReader(open(os.path.join(x["dir"], "predictions.csv")))
                     if row["true"] != row["pred"]]
            for row in sorted(wrong, key=lambda r: -float(r["confidence"])):
                rows.append([s, x["seed"], row["id"], row["speaker"], row["word"], row["true"], row["pred"],
                             row["confidence"], sp.get(row["id"], "")])
    write_csv(os.path.join(out, "failure_cases.csv"), rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="Results")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.results, "plots"), exist_ok=True)
    runs = load_runs(a.results)
    assert runs, "no finished runs found"
    accuracy_tables(runs, a.results)
    per_class(runs, a.results)
    confusions(runs, a.results)
    separability(runs, a.results)
    noise(runs, a.results)
    reliance(runs, a.results)
    gate(runs, a.results)
    curves(runs, a.results)
    layer_plot(runs, a.results)
    attention_plots(runs, a.results)
    failures(runs, a.results)
    print("tables and plots ->", a.results)


if __name__ == "__main__":
    main()
