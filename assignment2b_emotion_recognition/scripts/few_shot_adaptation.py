"""Cross-speaker error: wrong representation, or wrong labels on a good one?

The cross-speaker test clips of the new speaker form tight clusters in the speech model's temporal block, yet many
are classified wrongly. This script measures, on the saved test representations (reps.npz, no GPU needed):

  cluster purity   k-means with 7 clusters on the new speaker's vectors (no labels), each cluster then named with
                   the true labels by the best one-to-one match: how well the vectors separate the emotions, an
                   upper bound for any relabelling of the clusters
  few-shot         a logistic-regression classifier trained on k labelled clips per emotion of the new speaker
                   (k = 1, 2, 5, 10), tested on the rest of that speaker's clips; 20 random draws per k and seed
  zero-label       the same clusters, each named by the model's own predicted probabilities with one emotion per
                   cluster (Hungarian matching on the mean log-probability; TESS is balanced): adaptation with no
                   labels at all
  ensemble         the three seeds' predicted probabilities averaged (also for the unseen-word split)

    python scripts/few_shot_adaptation.py --results Results [--variants speech speech_spknorm fusion]
Writes Results/few_shot_adaptation.csv, Results/seed_ensemble.csv and Results/plots/few_shot_adaptation.png.
"""
import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.data import EMOTIONS  # noqa: E402

SPLITS = {"oaf2yaf": "OAF -> YAF", "yaf2oaf": "YAF -> OAF"}
BLOCK = {"speech": "temporal", "speech_spknorm": "temporal", "speech_logmel": "temporal", "fusion": "fusion",
         "speech_mean": "temporal", "speech_mean_aug": "temporal", "speech_logmel_aug": "temporal"}
KS = [1, 2, 5, 10]


def cluster_accuracy(x, y, seed=0):
    """Best one-to-one match of 7 k-means clusters to the 7 emotions (Hungarian), as accuracy."""
    c = KMeans(7, n_init=10, random_state=seed).fit_predict(x)
    m = np.zeros((7, 7))
    for ci, yi in zip(c, y):
        m[ci, yi] += 1
    r, col = linear_sum_assignment(-m)
    return m[r, col].sum() / len(y)


def zero_label(x, prob, seed=0):
    """Cluster the new speaker's vectors and name each cluster by the model's probabilities, one emotion each."""
    c = KMeans(7, n_init=20, random_state=seed).fit_predict(StandardScaler().fit_transform(x))
    score = np.array([[np.log(prob[c == k, e] + 1e-6).mean() for e in range(7)] for k in range(7)])
    r, col = linear_sum_assignment(-score)
    return np.array([dict(zip(r, col))[k] for k in c])


def probabilities(run):
    rows = list(csv.DictReader(open(os.path.join(run, "predictions.csv"))))
    return np.array([[float(r[f"p_{e}"]) for e in EMOTIONS] for r in rows]),         np.array([EMOTIONS.index(r["true"]) for r in rows])


def few_shot(x, y, k, draws=20, seed=0):
    rng = np.random.default_rng(seed)
    acc = []
    for _ in range(draws):
        tr = np.concatenate([rng.choice(np.where(y == c)[0], k, replace=False) for c in range(7)])
        te = np.setdiff1d(np.arange(len(y)), tr)
        sc = StandardScaler().fit(x[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(x[tr]), y[tr])
        acc.append((clf.predict(sc.transform(x[te])) == y[te]).mean())
    return float(np.mean(acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="Results")
    ap.add_argument("--variants", nargs="*", default=["speech", "speech_spknorm", "fusion"])
    a = ap.parse_args()
    rows = [["variant", "split", "seed", "model_accuracy", "cluster_purity"] + [f"few_shot_k{k}" for k in KS]
            + ["zero_label_adapted"]]
    for v in a.variants:
        for s in SPLITS:
            for seed in (0, 1, 2):
                run = os.path.join(a.results, "runs", v, s, f"seed{seed}")
                if not os.path.exists(os.path.join(run, "reps.npz")):
                    continue
                d = np.load(os.path.join(run, "reps.npz"))
                x, y = d[BLOCK[v]], d["labels"]
                acc = json.load(open(os.path.join(run, "metrics.json")))["accuracy"]
                prob, _ = probabilities(run)
                rows.append([v, s, seed, round(acc, 4), round(cluster_accuracy(x, y, seed), 4)]
                            + [round(few_shot(x, y, k, seed=seed), 4) for k in KS]
                            + [round(float((zero_label(x, prob, seed) == y).mean()), 4)])
                print(" | ".join(map(str, rows[-1])), flush=True)
    os.makedirs(os.path.join(a.results, "plots"), exist_ok=True)
    with open(os.path.join(a.results, "few_shot_adaptation.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    ens = [["variant", "split", "single_seed_mean", "three_seed_ensemble"]]
    for v in ["speech", "speech_logmel", "fusion", "fusion_xattn", "speech_mean", "speech_mean_aug", "speech_logmel_aug"]:
        for s in ["word"] + list(SPLITS):
            runs = [os.path.join(a.results, "runs", v, s, f"seed{k}") for k in (0, 1, 2)]
            if not all(os.path.exists(os.path.join(r, "predictions.csv")) for r in runs):
                continue
            ps = [probabilities(r) for r in runs]
            y = ps[0][1]
            ens.append([v, s, round(float(np.mean([(p.argmax(1) == y).mean() for p, _ in ps])), 4),
                        round(float((np.mean([p for p, _ in ps], 0).argmax(1) == y).mean()), 4)])
    with open(os.path.join(a.results, "seed_ensemble.csv"), "w", newline="") as f:
        csv.writer(f).writerows(ens)

    fig, axes = plt.subplots(1, len(SPLITS), figsize=(10, 3.6), sharey=True)
    for ax, (s, title) in zip(axes, SPLITS.items()):
        for v, c in zip(a.variants, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]):
            r = [row for row in rows[1:] if row[0] == v and row[1] == s]
            if not r:
                continue
            xs = [0] + KS
            ys = np.array([[row[3]] + row[5:5 + len(KS)] for row in r]) * 100
            ax.errorbar(xs, ys.mean(0), yerr=ys.std(0), marker="o", capsize=3, c=c, label=v)
            ax.axhline(100 * np.mean([row[4] for row in r]), c=c, ls=":", lw=1)
        ax.set_xticks([0] + KS, ["0\n(model)"] + [str(k) for k in KS])
        ax.set_xlabel("labelled clips per emotion from the new speaker")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("accuracy on the new speaker (%)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Adapting to an unseen speaker (dotted: cluster purity, k-means without labels, oracle naming)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.results, "plots", "few_shot_adaptation.png"), dpi=140)
    print("->", os.path.join(a.results, "few_shot_adaptation.csv"))


if __name__ == "__main__":
    main()
