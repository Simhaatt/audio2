"""Train and test every model variant on every split with several seeds.

    python scripts/run_all.py --features features --results Results [--seeds 0 1 2] [--epochs 40]

Variants (run directory Results/runs/<variant>/<split>/seed<k>):
  speech          WavLM features, BiLSTM + attention          (main speech model)
  speech_logmel   log-mel features, same temporal model      (feature ablation)
  text            BERT features, Transformer + attention     (main text model)
  fusion          speech + text encoders, gated fusion (GMU) (main multimodal model)
  fusion_concat   the same with plain concatenation           (fusion ablation)
  speech_spknorm  WavLM features normalised per speaker       (addition: unsupervised speaker normalisation)
  fusion_xattn    text tokens attend over speech frames       (addition: cross-modal attention fusion)
  speech_mean     mean of the 13 WavLM layers                 (control for the augmentation runs)
  speech_mean_aug the same, trained with 4 augmented copies of every training clip
  speech_logmel_aug  log-mel features, trained with the augmented copies
The last three need the features of scripts/extract_augmented.py and run only when named with --variants.

It calls the same functions as models/<pipeline>_pipeline/train.py and test.py, in one process, so the cached
features are loaded once.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.pipeline import test_main, train_main  # noqa: E402

VARIANTS = {"speech": ("speech", []), "speech_logmel": ("speech", ["--speech_features", "logmel"]),
            "speech_spknorm": ("speech", ["--speech_features", "wavlm_spknorm"]),
            "text": ("text", []), "fusion": ("fusion", []), "fusion_concat": ("fusion", ["--fusion", "concat"]),
            "fusion_xattn": ("fusion", ["--fusion", "xattn"]),
            # speaker augmentation (features from scripts/extract_augmented.py)
            "speech_mean": ("speech", ["--speech_features", "wavlmmean"]),
            "speech_mean_aug": ("speech", ["--speech_features", "wavlmmean", "--augment"]),
            "speech_logmel_aug": ("speech", ["--speech_features", "logmel", "--augment"])}
SPLITS = ["word", "oaf2yaf", "yaf2oaf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features")
    ap.add_argument("--results", default="Results")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--variants", nargs="*", default=[v for v in VARIANTS if "mean" not in v and "aug" not in v])
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    a = ap.parse_args()
    t0 = time.time()
    for v in a.variants:
        pipeline, extra = VARIANTS[v]
        for split in a.splits:
            for seed in a.seeds:
                run = os.path.join(a.results, "runs", v, split, f"seed{seed}")
                train_main(pipeline, ["--features", a.features, "--split", split, "--seed", str(seed), "--out", run,
                                      "--epochs", str(a.epochs), "--quiet"] + extra)
                test_main(pipeline, ["--features", a.features, "--run", run])
        print(f"== {v} done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
